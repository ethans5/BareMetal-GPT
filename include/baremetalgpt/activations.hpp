#pragma once

/// @file activations.hpp
/// @brief Activation functions for the BareMetal-GPT inference engine.
///
/// Currently provides:
///   - bmgpt::gelu(float)  — scalar GELU with tanh approximation
///   - bmgpt::gelu(Tensor) — element-wise GELU on a Tensor (any shape)
///
/// ## GELU variant: gelu_new (tanh approximation)
///
/// GPT-2 uses the tanh-approximated GELU introduced by Hendrycks & Gimpel (2016),
/// referred to as `gelu_new` in HuggingFace transformers:
///
///   GELU(x) ≈ 0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
///
/// This matches PyTorch's `torch.nn.functional.gelu(x, approximate='tanh')`
/// and the exact formula used in dump_reference.py.
///
/// **Not** the exact GELU (which uses erf): using erf here would produce
/// slightly different outputs than the PyTorch reference dumps.
///
/// Constants (double-precision literals, then cast to float at call site):
///   √(2/π)  ≈ 0.7978845608028654
///   0.044715  (exact, matches HuggingFace source)

#include "Tensor.hpp"

#include <cmath>

namespace bmgpt {

// ---------------------------------------------------------------------------
// Scalar GELU (tanh approximation) — inline for zero call overhead
// ---------------------------------------------------------------------------

/// @brief GELU activation with tanh approximation (gelu_new variant).
///
/// Formula:  0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))
///
/// @param x  Input scalar.
/// @return   GELU(x).
inline float gelu(float x) noexcept
{
    // Constants stored as double literals for precision, used as float.
    // kSqrt2OverPi = √(2/π)
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCoeff       = 0.044715f;

    float cube = x * x * x;
    float inner = kSqrt2OverPi * (x + kCoeff * cube);
    return 0.5f * x * (1.0f + std::tanh(inner));
}

// ---------------------------------------------------------------------------
// Element-wise GELU on a Tensor (any shape)
// ---------------------------------------------------------------------------

/// @brief Apply GELU element-wise to every element of a tensor.
///
/// @param x  Input tensor (any shape, any number of dimensions).
/// @return   A new tensor of the same shape with GELU applied to every element.
///           The input is unchanged (value semantics).
inline Tensor gelu(const Tensor& x)
{
    Tensor out(x.shape());

    const float* src = x.data();
    float*       dst = out.data();
    const size_t n   = x.numel();

    for (size_t i = 0; i < n; ++i) {
        dst[i] = gelu(src[i]);
    }

    return out;
}

} // namespace bmgpt
