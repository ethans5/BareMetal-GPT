# -*- coding: utf-8 -*-
"""
dump_reference.py — Phase 2, Deliverable 2

Runs a step-by-step GPT-2 forward pass and dumps every intermediate tensor
to .npy files for use as ground-truth values in C++ unit tests.

Strategy:
  Rather than using PyTorch hooks (which intercept but obscure the logic),
  this script manually steps through each sublayer, explicitly calling each
  module in order. This gives exact control over what is captured, and makes
  the dump logic easy to audit and extend.

For each Transformer block, the following intermediates are captured:
  block_N_in              : block input (= previous block output or initial embedding)
  block_N_ln1_out         : after pre-attention LayerNorm
  block_N_qkv_proj        : raw Q/K/V concatenated projection  [seq, 3*d_model]
  block_N_q               : Q split into heads               [n_heads, seq, head_dim]
  block_N_k               : K split into heads               [n_heads, seq, head_dim]
  block_N_v               : V split into heads               [n_heads, seq, head_dim]
  block_N_attn_scores_raw : Q @ K^T (unscaled, unmasked)     [n_heads, seq, seq]
  block_N_attn_scores_scaled : after division by sqrt(head_dim)
  block_N_attn_scores_masked : after causal mask applied
  block_N_attn_probs      : after softmax                    [n_heads, seq, seq]
  block_N_attn_weighted   : probs @ V                        [n_heads, seq, head_dim]
  block_N_attn_merged     : heads concatenated               [seq, d_model]
  block_N_attn_proj       : output projection                [seq, d_model]
  block_N_resid1          : first residual  (in + attn_proj)
  block_N_ln2_out         : after pre-MLP LayerNorm
  block_N_mlp_fc_out      : after MLP first linear (c_fc)
  block_N_mlp_gelu_out    : after GELU activation
  block_N_mlp_proj_out    : after MLP second linear (c_proj)
  block_N_out             : block output = resid1 + mlp_proj_out

Global tensors:
  input_ids               : token indices fed into the model
  token_embeddings        : wte(input_ids)
  position_embeddings     : wpe(position_ids)
  initial_hidden          : token_embeddings + position_embeddings
  final_ln_out            : output of the last LayerNorm (ln_f)
  logits                  : vocabulary projections  [seq, vocab_size]

Additionally, a metadata.json file is written that records the exact
configuration so the C++ tests can load dumps without hardcoding shapes.

Usage:
  python scripts/dump_reference.py              # dumps micro model (default)
  python scripts/dump_reference.py --full       # dumps full GPT-2 (slow)
  python scripts/dump_reference.py --both       # dumps both

Dependencies:
  pip install -r requirements.txt  (torch, transformers, safetensors, numpy)
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config

# Force UTF-8 on stdout/stderr (Windows legacy codepage safety)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.join(SCRIPT_DIR, "..")
WEIGHTS_DIR   = os.path.join(PROJECT_ROOT, "weights")
DUMPS_ROOT    = os.path.join(PROJECT_ROOT, "reference_dumps")

# ---------------------------------------------------------------------------
# 1. Micro-model config (must match export_weights.py exactly)
# ---------------------------------------------------------------------------

MICRO_CONFIG = GPT2Config(
    vocab_size          = 256,
    n_positions         = 64,
    n_embd              = 64,
    n_layer             = 2,
    n_head              = 2,
    n_inner             = 256,
    activation_function = "gelu_new",
    resid_pdrop         = 0.0,
    attn_pdrop          = 0.0,
    embd_pdrop          = 0.0,
    layer_norm_epsilon  = 1e-5,
    bos_token_id        = 1,
    eos_token_id        = 2,
)

# Fixed test input sequence for the micro model.
# Using a constant sequence (not random) ensures dumps are reproducible
# without needing to store the input elsewhere.
MICRO_INPUT_IDS = [42, 105, 12, 88]

# Fixed test input for full GPT-2.
# Token ids 198 = "\n", 15496 = "Hello", etc. — chosen to be valid GPT-2 tokens.
FULL_INPUT_IDS = [15496, 11, 616, 1438, 318, 402, 11571, 362]

# ---------------------------------------------------------------------------
# 2. GELU (tanh approximation) — must match GPT-2 exactly
# ---------------------------------------------------------------------------
# GPT-2 uses the tanh approximation variant, not torch.nn.functional.gelu
# (which defaults to the exact erf-based version since PyTorch 1.9).
# Replicating it here ensures our dump is computed the same way as the model.

def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """GPT-2 GELU: tanh approximation variant (gelu_new in HuggingFace)."""
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
    ))

# ---------------------------------------------------------------------------
# 3. Dump utilities
# ---------------------------------------------------------------------------

def save(dump_dir: str, name: str, tensor: torch.Tensor) -> None:
    """Saves a tensor as a .npy file. Shape and dtype are preserved."""
    arr = tensor.detach().float().cpu().numpy()
    path = os.path.join(dump_dir, f"{name}.npy")
    np.save(path, arr)
    print(f"  saved  {name:50s}  {str(list(tensor.shape)):25s}  {arr.dtype}")


def split_heads(tensor: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Reshapes [seq_len, d_model] -> [n_heads, seq_len, head_dim].
    This mirrors what GPT-2's attention does internally before computing scores.
    """
    seq_len, d_model = tensor.shape
    head_dim = d_model // n_heads
    # [seq, d_model] -> [seq, n_heads, head_dim] -> [n_heads, seq, head_dim]
    return tensor.view(seq_len, n_heads, head_dim).transpose(0, 1)

# ---------------------------------------------------------------------------
# 4. Manual step-by-step forward pass
# ---------------------------------------------------------------------------

def run_and_dump(model: GPT2Model, config: GPT2Config,
                 input_ids: list, dump_dir: str) -> None:
    """
    Executes the GPT-2 forward pass one sublayer at a time,
    saving every intermediate tensor as a .npy file.

    All operations are performed under torch.no_grad() and with the model
    in eval() mode to suppress dropout and batch-norm tracking.
    """
    os.makedirs(dump_dir, exist_ok=True)

    n_layers  = config.n_layer
    n_heads   = config.n_head
    d_model   = config.n_embd
    head_dim  = d_model // n_heads

    ids_tensor = torch.tensor([input_ids], dtype=torch.long)  # [1, seq_len]
    seq_len    = len(input_ids)

    print(f"\n  input_ids = {input_ids}  (seq_len={seq_len})")
    print(f"  n_layers={n_layers}, n_heads={n_heads}, d_model={d_model}, head_dim={head_dim}")
    print(f"\n  --- Global tensors ---")

    with torch.no_grad():
        # ----------------------------------------------------------------
        # Step 0: Embeddings
        # ----------------------------------------------------------------
        # GPT-2 embeds token ids and position ids separately, then sums them.
        # Position ids are always [0, 1, ..., seq_len-1].
        pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)  # [1, seq]

        tok_emb = model.wte(ids_tensor).squeeze(0)   # [seq, d_model]
        pos_emb = model.wpe(pos_ids).squeeze(0)       # [seq, d_model]
        hidden  = tok_emb + pos_emb                   # [seq, d_model]

        save(dump_dir, "input_ids",            ids_tensor.squeeze(0).float())
        save(dump_dir, "token_embeddings",     tok_emb)
        save(dump_dir, "position_embeddings",  pos_emb)
        save(dump_dir, "initial_hidden",       hidden)

        # ----------------------------------------------------------------
        # Steps 1..N: Transformer blocks
        # ----------------------------------------------------------------
        for i, block in enumerate(model.h):
            pfx = f"block_{i}"
            print(f"\n  --- Block {i} ---")

            save(dump_dir, f"{pfx}_in", hidden)

            # --- Pre-attention LayerNorm ---
            ln1_out = block.ln_1(hidden)             # [seq, d_model]
            save(dump_dir, f"{pfx}_ln1_out", ln1_out)

            # --- Q, K, V projection (single fused Conv1D in HuggingFace) ---
            # c_attn.weight is [d_model, 3*d_model] in HuggingFace (Conv1D convention).
            # We access it via the raw module to stay faithful to the model.
            qkv_proj = block.attn.c_attn(ln1_out)   # [seq, 3*d_model]
            save(dump_dir, f"{pfx}_qkv_proj", qkv_proj)

            # Split along last dim: each chunk is [seq, d_model]
            q_raw, k_raw, v_raw = qkv_proj.split(d_model, dim=-1)

            # Reshape into multi-head format: [n_heads, seq, head_dim]
            q = split_heads(q_raw, n_heads)
            k = split_heads(k_raw, n_heads)
            v = split_heads(v_raw, n_heads)

            save(dump_dir, f"{pfx}_q", q)
            save(dump_dir, f"{pfx}_k", k)
            save(dump_dir, f"{pfx}_v", v)

            # --- Scaled dot-product attention ---
            # Raw scores: Q @ K^T, shape [n_heads, seq, seq]
            scores_raw = torch.matmul(q, k.transpose(-2, -1))
            save(dump_dir, f"{pfx}_attn_scores_raw", scores_raw)

            # Scale by 1/sqrt(head_dim) to prevent softmax saturation
            scores_scaled = scores_raw / math.sqrt(head_dim)
            save(dump_dir, f"{pfx}_attn_scores_scaled", scores_scaled)

            # Causal mask: each position can only attend to earlier positions.
            # Shape: [1, seq, seq], with -inf above the diagonal.
            # We use float('-inf') so softmax maps those positions to 0.
            causal_mask = torch.full((seq_len, seq_len), float("-inf"))
            causal_mask = torch.triu(causal_mask, diagonal=1)  # upper triangle = -inf
            scores_masked = scores_scaled + causal_mask.unsqueeze(0)  # broadcast over heads
            save(dump_dir, f"{pfx}_attn_scores_masked", scores_masked)

            # Softmax over the key dimension (last dim)
            attn_probs = F.softmax(scores_masked, dim=-1)
            save(dump_dir, f"{pfx}_attn_probs", attn_probs)

            # Weighted sum over values: [n_heads, seq, head_dim]
            attn_weighted = torch.matmul(attn_probs, v)
            save(dump_dir, f"{pfx}_attn_weighted", attn_weighted)

            # Merge heads back: [n_heads, seq, head_dim] -> [seq, d_model]
            attn_merged = attn_weighted.transpose(0, 1).contiguous().view(seq_len, d_model)
            save(dump_dir, f"{pfx}_attn_merged", attn_merged)

            # Output projection
            attn_proj = block.attn.c_proj(attn_merged)   # [seq, d_model]
            save(dump_dir, f"{pfx}_attn_proj", attn_proj)

            # First residual connection
            resid1 = hidden + attn_proj
            save(dump_dir, f"{pfx}_resid1", resid1)

            # --- Pre-MLP LayerNorm ---
            ln2_out = block.ln_2(resid1)             # [seq, d_model]
            save(dump_dir, f"{pfx}_ln2_out", ln2_out)

            # --- MLP ---
            # c_fc: [d_model -> 4*d_model]
            mlp_fc_out = block.mlp.c_fc(ln2_out)    # [seq, 4*d_model]
            save(dump_dir, f"{pfx}_mlp_fc_out", mlp_fc_out)

            # GELU (tanh approximation — locked in Phase 0)
            mlp_gelu_out = gelu_new(mlp_fc_out)
            save(dump_dir, f"{pfx}_mlp_gelu_out", mlp_gelu_out)

            # c_proj: [4*d_model -> d_model]
            mlp_proj_out = block.mlp.c_proj(mlp_gelu_out)   # [seq, d_model]
            save(dump_dir, f"{pfx}_mlp_proj_out", mlp_proj_out)

            # Second residual connection — this is the block output
            hidden = resid1 + mlp_proj_out
            save(dump_dir, f"{pfx}_out", hidden)

        # ----------------------------------------------------------------
        # Final LayerNorm + Logits
        # ----------------------------------------------------------------
        print(f"\n  --- Final layers ---")

        final_ln_out = model.ln_f(hidden)            # [seq, d_model]
        save(dump_dir, "final_ln_out", final_ln_out)

        # GPT-2 uses weight tying: the lm_head matrix = wte.weight transposed.
        # logits[t, v] = dot(final_ln_out[t], wte.weight[v])
        # So: logits = final_ln_out @ wte.weight^T
        logits = final_ln_out @ model.wte.weight.t()  # [seq, vocab_size]
        save(dump_dir, "logits", logits)

        # ----------------------------------------------------------------
        # Sanity check: compare our logits to HuggingFace's output
        # ----------------------------------------------------------------
        hf_out    = model(ids_tensor, output_hidden_states=False)
        hf_hidden = hf_out.last_hidden_state.squeeze(0)  # [seq, d_model]
        hf_logits = hf_hidden @ model.wte.weight.t()

        max_diff = (logits - hf_logits).abs().max().item()
        if max_diff < 1e-4:
            print(f"\n  [OK] Sanity check: max diff vs HuggingFace = {max_diff:.2e}  (< 1e-4)")
        else:
            print(f"\n  [WARN] Sanity check: max diff = {max_diff:.2e}  -- investigate!")

# ---------------------------------------------------------------------------
# 5. Metadata file
# ---------------------------------------------------------------------------

def write_metadata(dump_dir: str, config: GPT2Config, input_ids: list) -> None:
    """
    Writes a JSON file describing the exact config and input used.
    C++ tests read this to know shapes without hardcoding them.
    """
    meta = {
        "input_ids":      input_ids,
        "seq_len":        len(input_ids),
        "n_layers":       config.n_layer,
        "n_heads":        config.n_head,
        "d_model":        config.n_embd,
        "head_dim":       config.n_embd // config.n_head,
        "vocab_size":     config.vocab_size,
        "context_length": config.n_positions,
        "d_ff":           config.n_inner if config.n_inner else 4 * config.n_embd,
        "layer_norm_eps": config.layer_norm_epsilon,
        "activation":     "gelu_new",
        "norm_order":     "pre",
        "description": (
            "Reference activations for BareMetal-GPT unit tests. "
            "All tensors are float32. "
            "C++ tests should match each dump to within 1e-4 absolute tolerance."
        ),
    }
    path = os.path.join(dump_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  [metadata] {path}")

# ---------------------------------------------------------------------------
# 6. Top-level dump runners
# ---------------------------------------------------------------------------

def dump_micro(seed: int = 42) -> None:
    """Dumps the micro test model (reproducible random weights)."""
    print("\n" + "=" * 60)
    print("Dumping: Micro Test Model")
    print("=" * 60)

    torch.manual_seed(seed)
    model = GPT2Model(MICRO_CONFIG)
    model.eval()

    dump_dir = os.path.join(DUMPS_ROOT, "micro")
    run_and_dump(model, MICRO_CONFIG, MICRO_INPUT_IDS, dump_dir)
    write_metadata(dump_dir, MICRO_CONFIG, MICRO_INPUT_IDS)

    print(f"\n  [DONE] {len(os.listdir(dump_dir))} files in {dump_dir}")


def dump_gpt2_full() -> None:
    """Dumps the full GPT-2 124M model (official HuggingFace weights)."""
    print("\n" + "=" * 60)
    print("Dumping: GPT-2 Small (124M)")
    print("=" * 60)
    print("  Loading official weights from HuggingFace (local cache if available)...")

    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    dump_dir = os.path.join(DUMPS_ROOT, "gpt2")
    run_and_dump(model, model.config, FULL_INPUT_IDS, dump_dir)
    write_metadata(dump_dir, model.config, FULL_INPUT_IDS)

    print(f"\n  [DONE] {len(os.listdir(dump_dir))} files in {dump_dir}")

# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dump GPT-2 forward-pass intermediates as .npy reference files."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--micro", action="store_true",
                       help="Dump micro test model only (default if no flag given)")
    group.add_argument("--full",  action="store_true",
                       help="Dump full GPT-2 124M only")
    group.add_argument("--both",  action="store_true",
                       help="Dump both micro and full GPT-2")
    args = parser.parse_args()

    print("BareMetal-GPT — Phase 2, Deliverable 2: Reference Dump")
    print("=" * 60)

    if args.full:
        dump_gpt2_full()
    elif args.both:
        dump_micro()
        dump_gpt2_full()
    else:
        # Default: micro only (fast, suitable for CI / unit test runs)
        dump_micro()

    print("\n" + "=" * 60)
    print("Deliverable 2 completed successfully.")
    print(f"Reference files in: {os.path.abspath(DUMPS_ROOT)}/")
    print("\nNext step: C++ WeightLoader (Deliverable 3)")
