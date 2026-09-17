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
| **2 — Weight Loading & Ref** | Export `.safetensors`, reference dumps, C++ `WeightLoader` | 🔧 In Progress |
| 3 — Layers | LayerNorm, GELU, Linear, Attention, FeedForward | ⬜ Planned |
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

30 unit tests powered by Google Test (100% passing):
- **23 Tensor tests** (`test_tensor`): memory layout, bounds checking, exceptions, 1D/2D/3D reshape & transpose.
- **7 MatMul tests** (`test_matmul`): square, non-square, row-matrix, identity, 1x1 edge cases, dimension mismatch & non-2D error checking.

---

## Phase 2 — Weights & Reference Tooling (In Progress 🔧)

Phase 2 builds the external bridge: exporting weights, establishing a surgical ground-truth debugging tool, and loading weights into C++ `Tensor` objects.

### Deliverable 1 — Weight Export (`scripts/export_weights.py`) ✅

- **HuggingFace Conv1D Transposition**: Standard PyTorch linear layers store weights in `[out_features, in_features]`, whereas HuggingFace GPT-2 `Conv1D` stores `[in_features, out_features]`. Weights are pre-transposed during export so the C++ engine consumes uniform `[out, in]` tensors.
- **Artifacts Generated (`weights/`)**:
  - `gpt2_weights.safetensors` (474.7 MB): 148 tensors for full GPT-2 124M.
  - `gpt2_config.json`: architectural hyperparameters.
  - `micro_gpt2_weights.safetensors` (473 KB): 28 tensors for reproducible unit testing (`d_model=64`, `n_layers=2`, `n_heads=2`, seed 42).
  - `micro_gpt2_config.json`: micro model configuration.
- **Post-export Round-Trip Verification**: Automatic reload check confirming tensor counts, `float32` precision, and contiguous memory layout.

### Upcoming Deliverables

- **Deliverable 2**: `scripts/dump_reference.py` — forward pass dump of intermediate activations (`.npy`) for exact layer-by-layer C++ numerical validation.
- **Deliverable 3**: C++ `WeightLoader` — parser for `.safetensors` format populating `Tensor` instances.

---

## Setup & Development

### 1. Python Environment (Weights & Tooling)

We isolate all Python tooling inside a local virtual environment:

```powershell
# Create virtual environment
python -m venv .venv

# Activate environment (PowerShell)
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Export model weights (.safetensors)
python scripts/export_weights.py
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
│   └── matmul.hpp                        # Matrix multiplication function declaration
├── src/
│   ├── Tensor.cpp                        # Tensor implementation (at, reshape, transpose)
│   └── matmul.cpp                        # matmul implementation (cache-friendly i-k-j)
├── scripts/
│   └── export_weights.py                 # Weight exporter (GPT-2 & micro model to .safetensors)
├── tests/
│   ├── CMakeLists.txt                    # GTest v1.14.0 via FetchContent
│   ├── test_tensor.cpp                   # 23 unit tests for Tensor
│   └── test_matmul.cpp                   # 7 unit tests for matmul
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

