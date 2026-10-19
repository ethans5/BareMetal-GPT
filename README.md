# BareMetal-GPT

> A **from-scratch C++17 inference engine** for GPT-2 (124M parameters) — no PyTorch, no ONNX, no external ML framework.  
> Built as a deep-learning systems programming project to demonstrate low-level mastery of transformer internals.

---

## Overview

BareMetal-GPT loads official GPT-2 Small weights and runs autoregressive text generation entirely in C++.  
The goal is not to compete with optimized runtimes, but to **understand every layer** by implementing it from raw math.

### Two-Model Strategy

| Model | Purpose | Typical dimensions |
|---|---|---|
| **Micro test model** | Unit tests & per-phase validation (fast iteration) | `d_model=64`, `n_layers=2`, `n_heads=2` |
| **GPT-2 124M** | Final demo — coherent text generation | `d_model=768`, `n_layers=12`, `n_heads=12` |

Both models are **structurally identical** (same pre-norm order, same GELU variant, same attention formula, same learned positional embeddings). Only the numeric dimensions differ. This ensures that passing all tests on the micro model **guarantees correctness** on full GPT-2.

---

## GPT-2 Small — Architecture Summary

| Parameter | Value |
|---|---|
| Layers (`n_layers`) | 12 |
| Attention heads (`n_heads`) | 12 |
| Embedding dimension (`d_model`) | 768 |
| Vocabulary size | 50 257 |
| Max sequence length | 1 024 |

### Key design decisions (locked in Phase 0)

- **Normalization**: Pre-norm — `LayerNorm` is applied *before* attention and *before* the feed-forward block, with a residual connection on the un-normalized input.
  ```
  x = x + Attention(LayerNorm(x))
  x = x + FeedForward(LayerNorm(x))
  ```
  A final `LayerNorm` is applied after the last block, before the vocabulary projection.

- **Activation**: GELU with the **tanh approximation** (`gelu_new` in HuggingFace):
  ```
  GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
  ```

- **Positional encoding**: Learned positional embeddings — a `[1024, 768]` weight matrix indexed by token position, added to the token embeddings. No sinusoidal, no RoPE.

- **Tokenizer**: Byte-level BPE using the official GPT-2 `vocab.json` and `merges.txt`.

- **Weight format**: `.safetensors` exported from HuggingFace `transformers`.

---

## Project Roadmap

| Phase | Description | Status |
|---|---|---|
| **0 — Scoping** | Lock architecture, dimensions, activation, norm order | ✅ Done |
| **1 — Tensor** | Core `Tensor` class, element access, reshape, transpose, matmul | ✅ Done |
| **2 — Weight Loading & Ref** | Export `.safetensors`, reference dumps, C++ `WeightLoader` | ✅ Done |
| **3 — Layers** | LayerNorm, GELU (tanh), Linear, MLP, CausalSelfAttention | ✅ Done |
| 4 — Transformer | Assemble full GPT-2 forward pass | ⬜ Planned |
| 5 — Tokenizer & CLI | BPE tokenizer + `ChatCLI` demo + KV-Cache | ⬜ Planned |
| 6 — Performance | Cache tiling & multithreading | ⬜ Planned |

---

## Phase 1 — Tensor (Completed ✅)

The `Tensor` class and matrix multiplication engine are the computational foundation of the entire inference system. Every weight matrix, activation, and intermediate state flows through them.

### What's implemented

- **N-dimensional storage** — flat `std::vector<float>` with row-major (C-contiguous) layout.
- **Precomputed strides** — $O(D)$ index flattening with precomputed stride vectors.
- **Strict bounds-checking** — `at({i, j, ...})` with explicit `std::out_of_range` exceptions.
- **Arithmetic operations** — element-wise `operator+` on matching shapes.
- **`reshape()`** — changes view with strict element-count invariant verification ($\prod \text{old} == \prod \text{new}$).
- **`transpose(dim0, dim1)`** — generalized N-D dimension swapping using an odometer iterator with guaranteed physical C-contiguity.
- **`bmgpt::matmul()`** — free function for 2D matrix multiplication with cache-optimized $i, k, j$ loop ordering.
- **Zero-warning build** — compiled with `-Wall -Wextra -Wpedantic -Werror` (and `/W4 /WX` on MSVC).

### Testing & Validation

30 unit tests powered by Google Test:
- **23 Tensor tests** (`test_tensor`): memory layout, bounds checking, exceptions, 1D/2D/3D reshape & transpose.
- **7 MatMul tests** (`test_matmul`): square, non-square, row-matrix, identity, 1x1 edge cases, dimension mismatch & non-2D error checking.

---

## Phase 2 — Weights & Reference Tooling (Completed ✅)

Phase 2 builds the external bridge between Python/HuggingFace and the C++ engine: exporting weights, producing surgical ground-truth dumps for every intermediate tensor, and the C++ `WeightLoader` that ingests them.

### Deliverable 1 — Weight Export (`scripts/export_weights.py`) ✅

- **HuggingFace Conv1D Transposition**: HuggingFace GPT-2 uses custom `Conv1D` layers that store weights as `[in, out]`, opposite of the standard `[out, in]` convention. Weights are pre-transposed at export time so the C++ engine always receives uniform `[out, in]` tensors — no special casing needed in the loader.
- **Artifacts generated in `weights/`**:
  - `gpt2_weights.safetensors` (474.7 MB) — 148 tensors, full GPT-2 124M.
  - `gpt2_config.json` — architectural hyperparameters.
  - `micro_gpt2_weights.safetensors` (473 KB) — 28 tensors, reproducible micro model (`d_model=64`, `n_layers=2`, `n_heads=2`, seed 42).
  - `micro_gpt2_config.json` — micro model configuration.
- **Round-trip verification**: automatic reload after export confirms tensor counts, `float32` dtype, and physical contiguity.

### Deliverable 2 — Reference Dumps (`scripts/dump_reference.py`) ✅

Runs a manual, step-by-step GPT-2 forward pass and saves every intermediate tensor as a `.npy` file. These are the ground-truth values used in all C++ unit tests from Phase 3 onwards.

- **Strategy**: instead of PyTorch hooks, the forward pass is implemented by hand, calling each sublayer in order. This captures everything — including intermediate steps *inside* the attention block (unscaled scores, causal mask, post-softmax probabilities, per-head weighted sums) that hooks cannot reach.
- **45 files generated in `reference_dumps/micro/`** for the fixed input `[42, 105, 12, 88]`:

  | Tensor | Shape | Meaning |
  |---|---|---|
  | `token_embeddings` | `[4, 64]` | `wte(input_ids)` |
  | `position_embeddings` | `[4, 64]` | `wpe([0,1,2,3])` |
  | `initial_hidden` | `[4, 64]` | sum of both embeddings |
  | `block_N_ln1_out` | `[4, 64]` | pre-attention LayerNorm |
  | `block_N_q/k/v` | `[2, 4, 32]` | Q, K, V per head |
  | `block_N_attn_probs` | `[2, 4, 4]` | attention weights after softmax |
  | `block_N_mlp_gelu_out` | `[4, 256]` | GELU activation (tanh approx) |
  | `logits` | `[4, 256]` | vocabulary projections |
  | `metadata.json` | — | config + input_ids for C++ loaders |

- **Sanity check**: manual logits vs. HuggingFace output — max absolute diff = **1.49e-07** (threshold: 1e-4 ✅).
- **Usage**:
  ```powershell
  python scripts/dump_reference.py          # micro model (default, fast)
  python scripts/dump_reference.py --full   # full GPT-2 124M
  python scripts/dump_reference.py --both   # both
  ```

### Deliverable 3 — C++ WeightLoader (`include/baremetalgpt/WeightLoader.hpp`) ✅

A zero-dependency `.safetensors` parser written entirely in C++17 stdlib — no external JSON library.

**Binary format parsed**:
```
[8 bytes]  uint64_le        — byte length of the JSON header (N)
[N bytes]  UTF-8 JSON       — map: name → {dtype, shape, data_offsets:[begin, end]}
[blob]     raw float32 data — tensor buffers concatenated
```
`data_offsets` are relative to the blob start (byte 8+N), not the file start.

**API**:
```cpp
bmgpt::WeightLoader loader("weights/micro_gpt2_weights.safetensors");
Tensor wte = loader.load("wte.weight");   // shape [256, 64]
bool ok     = loader.has("ln_f.bias");    // true
size_t n    = loader.num_tensors();       // 28
```

**Design decisions**:
- **Lazy I/O**: header parsed once at construction; each `load()` opens, seeks, reads, closes — no persistent file descriptor.
- **Strict validation**: dtype must be `F32`; byte range verified against `numel * sizeof(float)` at parse time.
- **Error propagation**: every failure path throws a descriptive `std::runtime_error`.
- **14 unit tests** in `tests/test_weight_loader.cpp`: tensor count, key lookup, full 28-key enumeration, shape checks for all layer types, numeric round-trip (tolerance 1e-6), and error handling (missing file, unknown key).

---

## Phase 3 — Layers & Attention (Completed ✅)

Phase 3 builds the five computational building blocks that form the heart of the Transformer architecture, each numerically verified against the Phase 2 ground-truth dumps with strict precision (`max_abs_diff < 1e-4`).

### What's implemented

1. **`LayerNorm` (`include/baremetalgpt/LayerNorm.hpp`, `src/LayerNorm.cpp`)**
   - Implements GPT-2 pre-norm layer normalization:
     $$y = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
   - **Numerical stability**: two-pass calculation of mean and variance to prevent catastrophic cancellation, with $\epsilon = 10^{-5}$ matching PyTorch.
   - **Row-wise normalization**: token activations $[T, d_{model}]$ are normalized independently per token row. Reciprocal standard deviation $(1 / \sqrt{\sigma^2 + \epsilon})$ is computed once per row.
   - **Validation**: verified against `block_0_ln1_out.npy`, `block_0_ln2_out.npy`, and `final_ln_out.npy` (max error $< 2.4 \times 10^{-7}$).

2. **`GELU` (`include/baremetalgpt/activations.hpp`)**
   - Exact `gelu_new` tanh approximation variant matching HuggingFace/PyTorch:
     $$\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot \left(x + 0.044715 \cdot x^3\right)\right)\right)$$
   - **Zero overhead**: inline scalar implementation using double-precision constants ($\sqrt{2/\pi} \approx 0.7978845608028654$) and element-wise `Tensor` overload preserving value semantics.
   - **Validation**: verified against analytical test points and intermediate dumps `block_N_mlp_gelu_out.npy` (max error $< 1.2 \times 10^{-7}$).

3. **`Linear` (`include/baremetalgpt/Linear.hpp`, `src/Linear.cpp`)**
   - Affine transformation layer: $y = x \cdot W^T + b$.
   - **Memory locality**: $W$ is stored in standard `[out, in]` format and physically transposed to `[in, out]` prior to `matmul`, ensuring optimal cache-line streaming in the $i, k, j$ multiplication kernel.
   - **Broadcast addition**: bias vector is added row-wise across all sequence positions.
   - **Validation**: verified across all 4 linear projections (`c_attn`, `c_proj`, `c_fc`, `c_proj`) against reference dumps (max error $< 3.5 \times 10^{-7}$).

4. **`FeedForward` / MLP (`include/baremetalgpt/FeedForward.hpp`, `src/FeedForward.cpp`)**
   - Two-layer feed-forward network:
     $$\text{FFN}(x) = \text{Linear}_{\text{proj}}(\text{GELU}(\text{Linear}_{\text{fc}}(x)))$$
   - Projects $d_{model} \to d_{ff}$ (4x expansion), applies GELU, and contracts $d_{ff} \to d_{model}$.
   - Construction supported via direct `WeightLoader` prefix (e.g. `"h.0.mlp"`) or explicit layer injection.
   - **Validation**: step-by-step and full forward pass verified against `block_N_mlp_proj_out.npy` (max error $< 3.8 \times 10^{-7}$).

5. **`CausalSelfAttention` (`include/baremetalgpt/CausalSelfAttention.hpp`, `src/CausalSelfAttention.cpp`)**
   - Multi-head causal self-attention with scaled dot-product:
     1. **Fused QKV projection**: single forward pass $[T, d_{model}] \to [T, 3 \cdot d_{model}]$ via `c_attn`.
     2. **Head splitting**: sliced into $H$ heads of dimension $d_{head} = d_{model} / H$.
     3. **Scaled dot-product**: $(Q_h \cdot K_h^T) / \sqrt{d_{head}}$.
     4. **Causal masking**: upper-triangular masking ($j > i$) prevents tokens from attending to the future.
     5. **Stable softmax**: max-subtracted softmax prevents float32 overflow.
     6. **Weighted accumulation**: attention probabilities multiplied by $V_h$.
     7. **Head concatenation & projection**: merged heads $[T, d_{model}]$ projected via `c_proj`.
   - **Causality invariant**: strictly verified that token $0$ output in $[x_0, x_1, \dots]$ matches $[x_0]$ alone.
   - **Validation**: verified against `block_0_attn_proj.npy` and `block_1_attn_proj.npy` (max error $< 4.1 \times 10^{-7}$).

### Testing & Validation Summary

**110 unit tests** powered by Google Test (**100% passing**, zero compiler warnings):
- **23 Tensor tests** (`test_tensor`): memory, indexing, bounds, reshape, transpose.
- **7 MatMul tests** (`test_matmul`): 2D algebra, shapes, cache-friendly kernel correctness.
- **14 WeightLoader tests** (`test_weight_loader`): header parsing, key discovery, round-trip checks.
- **13 LayerNorm tests** (`test_layer_norm`): normalization statistics, error paths, numerical dumps match.
- **18 Activation tests** (`test_activations`): scalar analytical points, tensor preservation, GELU dumps match.
- **16 Linear tests** (`test_linear`): shapes, bias broadcasting, mathematical limits, projection dumps match.
- **8 FeedForward tests** (`test_feedforward`): dimensional flow, sublayer validation, full MLP match.
- **11 CausalSelfAttention tests** (`test_causal_self_attention`): head division, causality invariant, attention dumps match.

---

## Setup & Development

### 1. Python Environment (Weights & Tooling)

We isolate all Python tooling inside a local virtual environment:

```powershell
# Create virtual environment (one time)
python -m venv .venv

# Activate environment (every session)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Deliverable 1 — Export GPT-2 weights to .safetensors
python scripts/export_weights.py

# Deliverable 2 — Dump forward-pass reference tensors to .npy
python scripts/dump_reference.py           # micro model (fast)
python scripts/dump_reference.py --full    # full GPT-2 124M
```

### 2. C++ Build & Tests

```bash
# Configure (downloads Google Test automatically via FetchContent)
cmake -S . -B build

# Build library + test targets
cmake --build build

# Run all test suites
ctest --test-dir build --output-on-failure
```

---

## Project Structure

```
BareMetal-GPT/
├── CMakeLists.txt                        # Root build — lib + strict compiler flags
├── requirements.txt                      # Python dependencies (torch, transformers, safetensors)
├── include/baremetalgpt/
│   ├── Tensor.hpp                        # Tensor class declaration
│   ├── matmul.hpp                        # Matrix multiplication function declaration
│   ├── WeightLoader.hpp                  # .safetensors parser declaration
│   ├── LayerNorm.hpp                     # Layer normalization declaration
│   ├── activations.hpp                   # GELU activation functions (scalar & tensor)
│   ├── Linear.hpp                        # Fully-connected linear layer declaration
│   ├── FeedForward.hpp                   # Two-layer MLP declaration
│   └── CausalSelfAttention.hpp           # Multi-head causal self-attention declaration
├── src/
│   ├── Tensor.cpp                        # Tensor implementation (at, reshape, transpose)
│   ├── matmul.cpp                        # matmul implementation (cache-friendly i-k-j)
│   ├── WeightLoader.cpp                  # .safetensors parser (zero-dependency, stdlib only)
│   ├── LayerNorm.cpp                     # Two-pass LayerNorm implementation
│   ├── Linear.cpp                        # Linear layer with transposed weight & bias broadcast
│   ├── FeedForward.cpp                   # FeedForward (c_fc -> GELU -> c_proj) implementation
│   └── CausalSelfAttention.cpp           # Multi-head attention with causal masking & softmax
├── scripts/
│   ├── export_weights.py                 # Deliverable 1: export GPT-2 & micro weights to .safetensors
│   └── dump_reference.py                 # Deliverable 2: dump all forward-pass intermediates to .npy
├── tests/
│   ├── CMakeLists.txt                    # GTest v1.14.0 via FetchContent
│   ├── test_tensor.cpp                   # 23 unit tests for Tensor
│   ├── test_matmul.cpp                   # 7 unit tests for matmul
│   ├── test_weight_loader.cpp            # 14 unit tests for WeightLoader
│   ├── test_layer_norm.cpp               # 13 unit tests for LayerNorm
│   ├── test_activations.cpp              # 18 unit tests for GELU
│   ├── test_linear.cpp                   # 16 unit tests for Linear
│   ├── test_feedforward.cpp              # 8 unit tests for FeedForward / MLP
│   └── test_causal_self_attention.cpp    # 11 unit tests for CausalSelfAttention
└── README.md                             # Project overview and documentation
```

---

## Requirements

- **C++17** compiler (GCC ≥ 9, Clang ≥ 10, MSVC ≥ 19.14)
- **CMake ≥ 3.20**
- **Python ≥ 3.10** (for weight export & reference dumps)
- Internet connection on first build/export (fetches Google Test and HuggingFace weights)

---

## License

This is a personal learning project. No license has been chosen yet.

