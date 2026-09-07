# BareMetal-GPT

> A **from-scratch C++17 inference engine** for GPT-2 (124M parameters) — no PyTorch, no ONNX, no external ML framework.  
> Built as a deep-learning systems programming project to demonstrate low-level mastery of transformer internals.

---

## Overview

BareMetal-GPT loads the official GPT-2 Small weights and runs autoregressive text generation entirely in C++.  
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

## Project Phases

| Phase | Description | Status |
|---|---|---|
| **0 — Scoping** | Lock architecture, dimensions, activation, norm order | ✅ Done |
| **1 — Tensor** | Core `Tensor` class, element access, reshape, transpose, matmul | ✅ Done |
| **2 — Weight loading** | Parse `.safetensors`, load into Tensors | 🔧 Next |
| 3 — Layers | LayerNorm, GELU, Linear, Attention, FeedForward | ⬜ Planned |
| 4 — Transformer | Assemble full GPT-2 forward pass | ⬜ Planned |
| 5 — Tokenizer & CLI | BPE tokenizer + `ChatCLI` demo | ⬜ Planned |

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

### Building & Running Tests

```bash
# Configure (downloads Google Test automatically via FetchContent)
cmake -S . -B build

# Build library + test targets
cmake --build build

# Run all test suites
ctest --test-dir build --output-on-failure
```

### Project structure

```
BareMetal-GPT/
├── CMakeLists.txt                        # Root build — lib + strict compiler flags
├── include/baremetalgpt/
│   ├── Tensor.hpp                        # Tensor class declaration
│   └── matmul.hpp                        # Matrix multiplication function declaration
├── src/
│   ├── Tensor.cpp                        # Tensor implementation (at, reshape, transpose)
│   └── matmul.cpp                        # matmul implementation (cache-friendly i-k-j)
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
- Internet connection on first build (to fetch Google Test)

---

## License

This is a personal learning project. No license has been chosen yet.
