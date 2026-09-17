# -*- coding: utf-8 -*-
"""
export_weights.py — Phase 2, Deliverable 1

Exports GPT-2 (Small, 124M) and a micro test model weights
into .safetensors format, ready to be ingested by the C++ WeightLoader.

Two sets of artifacts generated in weights/:
  - gpt2_weights.safetensors       : full 124M GPT-2 weights
  - gpt2_config.json               : model hyperparameters
  - micro_gpt2_weights.safetensors : micro model (n_layers=2, d_model=64, n_heads=2)
  - micro_gpt2_config.json         : micro model hyperparameters

Dependencies:
  pip install transformers safetensors torch

Usage:
  python scripts/export_weights.py
"""

import os
import sys
import json
import torch

# Force UTF-8 on stdout/stderr to avoid UnicodeEncodeError on Windows
# (terminals might use legacy codepages like cp1252/cp1255)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from transformers import GPT2Model, GPT2Config
from safetensors.torch import save_file
from safetensors import safe_open

# ---------------------------------------------------------------------------
# 0. Output directory
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Tensor extraction and normalization
# ---------------------------------------------------------------------------
# In HuggingFace, GPT-2 uses custom "Conv1D" layers rather than standard nn.Linear.
# A Conv1D layer stores weights as [in_features, out_features], which is the
# opposite of the standard [out_features, in_features] layout.
#
# Why this matters for our C++ engine:
#   The C++ WeightLoader and Linear layers expect W with shape [out, in] to compute
#   y = x @ W^T + b (standard Linear layer convention).
#   If we kept raw Conv1D orientations, C++ would need to track and transpose them.
#   Transposing them once during export ensures a uniform convention across all layers.

CONV1D_WEIGHT_SUFFIXES = ("c_attn.weight", "c_proj.weight", "c_fc.weight")

def extract_state_dict(model: GPT2Model) -> dict:
    """
    Extracts all model tensors and normalizes Conv1D weights
    from [in, out] to [out, in].
    Returns a dict {tensor_key: torch.Tensor}, all as contiguous float32.
    """
    tensors = {}
    for name, tensor in model.state_dict().items():
        # GPT2Model prefixes keys with "transformer." — strip it
        # for clean names: "wte.weight", "h.0.ln_1.weight", etc.
        key = name.removeprefix("transformer.")

        # Transpose Conv1D weights: [in, out] -> [out, in]
        if any(key.endswith(suffix) for suffix in CONV1D_WEIGHT_SUFFIXES):
            tensor = tensor.t().contiguous()

        # safetensors requires contiguous float32 buffers
        tensors[key] = tensor.to(dtype=torch.float32).contiguous()

    return tensors


def export_config(config: GPT2Config, output_path: str) -> None:
    """
    Saves model hyperparameters into a JSON file.
    The C++ WeightLoader inspects this file to allocate correct Tensor dimensions.
    """
    meta = {
        "n_layers":       config.n_layer,
        "n_heads":        config.n_head,
        "d_model":        config.n_embd,
        "vocab_size":     config.vocab_size,
        "context_length": config.n_positions,
        "d_ff":           config.n_inner if config.n_inner else 4 * config.n_embd,
        "activation":     "gelu_new",   # tanh approximation — locked in Phase 0
        "norm_order":     "pre",        # pre-norm — locked in Phase 0
        "pos_encoding":   "learned",    # learned positional embeddings — locked in Phase 0
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  [config] {output_path}")


# ---------------------------------------------------------------------------
# 2. Export official GPT-2 Small (124M) weights
# ---------------------------------------------------------------------------

def export_gpt2_full() -> int:
    """Exports official GPT-2 124M weights and returns total exported tensor count."""
    print("\n=== GPT-2 Small (124M) ===")
    print("  Loading official weights from HuggingFace (local cache if available)...")

    # GPT2Model represents the transformer backbone without the lm_head language head.
    # In GPT-2, lm_head shares its weights directly with wte (weight tying).
    # The C++ engine reuses wte.weight for vocabulary projection.
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()

    tensors = extract_state_dict(model)

    # Sanity check: ensure fundamental keys are present
    required = ["wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"]
    for k in required:
        assert k in tensors, f"Missing critical key in export: {k}"

    print(f"  {len(tensors)} tensors extracted.")
    _print_tensor_table(tensors)

    out_path = os.path.join(OUTPUT_DIR, "gpt2_weights.safetensors")
    save_file(tensors, out_path)
    size_mb = os.path.getsize(out_path) / (1024 ** 2)
    print(f"\n  [OK] {out_path}  ({size_mb:.1f} MB)")

    export_config(model.config, os.path.join(OUTPUT_DIR, "gpt2_config.json"))
    return len(tensors)


# ---------------------------------------------------------------------------
# 3. Export micro test model
# ---------------------------------------------------------------------------
# Structurally identical to GPT-2, but with scaled-down dimensions.
# Initialized with a fixed random seed for reproducible numeric comparisons.
# C++ unit tests compare against exact values computed from these weights.
#
# Phase 0 constraint: identical activation (gelu_new), pre-norm ordering,
# and learned positional embeddings — only numeric dimensions differ.

MICRO_CONFIG = GPT2Config(
    vocab_size          = 256,     # scaled down (vs 50257)
    n_positions         = 64,      # max context (vs 1024)
    n_embd              = 64,      # d_model (vs 768)
    n_layer             = 2,       # number of transformer blocks (vs 12)
    n_head              = 2,       # attention heads (vs 12)
    n_inner             = 256,     # d_ff = 4 * d_model (same ratio as GPT-2)
    activation_function = "gelu_new",
    resid_pdrop         = 0.0,     # no dropout during inference
    attn_pdrop          = 0.0,
    embd_pdrop          = 0.0,
    layer_norm_epsilon  = 1e-5,    # identical to GPT-2
    # Ensure bos/eos tokens fall within the reduced vocabulary range [0, 255]
    bos_token_id        = 1,
    eos_token_id        = 2,
)

def export_micro_model() -> int:
    """Exports micro model weights and returns total exported tensor count."""
    print("\n=== Micro Test Model ===")
    print("  Dimensions: n_layers=2, n_heads=2, d_model=64, vocab=256, ctx=64")

    torch.manual_seed(42)   # fixed seed for reproducible weights
    model = GPT2Model(MICRO_CONFIG)
    model.eval()

    tensors = extract_state_dict(model)
    print(f"  {len(tensors)} tensors extracted.")
    _print_tensor_table(tensors)

    out_path = os.path.join(OUTPUT_DIR, "micro_gpt2_weights.safetensors")
    save_file(tensors, out_path)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n  [OK] {out_path}  ({size_kb:.1f} KB)")

    export_config(MICRO_CONFIG, os.path.join(OUTPUT_DIR, "micro_gpt2_config.json"))
    return len(tensors)


# ---------------------------------------------------------------------------
# 4. Post-export round-trip verification
# ---------------------------------------------------------------------------
# Re-reads exported .safetensors files to verify:
#   a) number of tensors matches export count
#   b) all buffers are contiguous float32
# This serves as a smoke test before feeding weights into C++.

def verify_export(filename: str, expected_count: int) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    print(f"\n  Verifying {filename} ...")

    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        assert len(keys) == expected_count, \
            f"Expected {expected_count} tensors, found {len(keys)}"
        for key in keys:
            t = f.get_tensor(key)
            assert t.dtype == torch.float32, f"{key}: unexpected dtype {t.dtype} (expected float32)"
            assert t.is_contiguous(),        f"{key}: non-contiguous buffer after reload"

    print(f"  [OK] {len(keys)} tensors, all contiguous float32.")


# ---------------------------------------------------------------------------
# 5. Tensor table formatting helper
# ---------------------------------------------------------------------------

def _print_tensor_table(tensors: dict) -> None:
    print("  Tensor breakdown:")
    for name, t in tensors.items():
        shape_str = str(list(t.shape))
        print(f"    {name:<50s}  {shape_str:<25s}  dtype={t.dtype}")


# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("BareMetal-GPT — Phase 2, Deliverable 1: Weight Export")
    print("=" * 60)

    n_full  = export_gpt2_full()
    n_micro = export_micro_model()

    print("\n=== Post-Export Round-Trip Verification ===")
    verify_export("gpt2_weights.safetensors",       n_full)
    verify_export("micro_gpt2_weights.safetensors", n_micro)

    print("\n" + "=" * 60)
    print("Deliverable 1 completed successfully.")
    print(f"Generated files in: {os.path.abspath(OUTPUT_DIR)}/")
    print("  gpt2_weights.safetensors       — GPT-2 124M weights")
    print("  gpt2_config.json")
    print("  micro_gpt2_weights.safetensors — micro test model weights")
    print("  micro_gpt2_config.json")
    print("\nNext step: scripts/dump_reference.py (Deliverable 2)")

