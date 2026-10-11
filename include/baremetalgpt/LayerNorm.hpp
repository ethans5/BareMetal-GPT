#pragma once

/// @file LayerNorm.hpp
/// @brief Layer Normalization for the BareMetal-GPT inference engine.
///
/// Implements the pre-norm variant used in GPT-2:
///   y = γ · (x - μ) / √(σ² + ε) + β
///
/// where:
///   - μ  = mean of x over the last dimension (d_model)
///   - σ² = variance of x over the last dimension
///   - γ  (weight) and β (bias) are learned 1-D parameters of size d_model
///   - ε  = 1e-5 (numerical stability constant, matching PyTorch default)
///
/// Normalization is applied **row-wise**: each token position is normalized
/// independently using its own mean and variance.  This matches PyTorch's
/// `nn.LayerNorm(d_model)` behaviour applied to a tensor of shape [T, d_model].
///
/// Usage:
/// @code
///   bmgpt::WeightLoader loader("weights/micro_gpt2_weights.safetensors");
///   bmgpt::LayerNorm ln(loader, "h.0.ln_1");   // loads weight + bias
///   Tensor out = ln.forward(x);                // x: [T, d_model]
/// @endcode

#include "Tensor.hpp"
#include "WeightLoader.hpp"

#include <string>

namespace bmgpt {

/// @class LayerNorm
/// @brief Layer normalization with learned affine parameters γ and β.
///
/// Weights are loaded once at construction from a WeightLoader. The forward
/// pass is stateless and thread-safe (no mutable state after construction).
class LayerNorm {
public:
    // ── Constructor ─────────────────────────────────────────────────

    /// @brief Load γ (weight) and β (bias) from a .safetensors file.
    /// @param loader   An already-opened WeightLoader pointing at the weight file.
    /// @param prefix   Key prefix in the file, e.g. "h.0.ln_1" or "ln_f".
    ///                 The constructor looks up "<prefix>.weight" and "<prefix>.bias".
    /// @throws std::runtime_error if either key is missing from the loader,
    ///         or if weight/bias shapes are inconsistent (weight.numel() != bias.numel()).
    LayerNorm(const WeightLoader& loader, const std::string& prefix);

    // ── Forward pass ─────────────────────────────────────────────────

    /// @brief Apply layer normalization to the input tensor.
    /// @param x  Input tensor of shape [T, d_model].  The last dimension must
    ///           match d_model (= weight_.numel()).  T can be any positive value.
    /// @return   A new tensor of the same shape [T, d_model] with each row
    ///           normalized then scaled and shifted by γ and β.
    /// @throws std::invalid_argument if x.ndim() != 2 or x.size(1) != d_model().
    Tensor forward(const Tensor& x) const;

    // ── Metadata ─────────────────────────────────────────────────────

    /// @brief The normalized dimension size (= number of elements in γ and β).
    size_t d_model() const noexcept { return weight_.numel(); }

    /// @brief Read-only access to the γ weight vector (shape [d_model]).
    const Tensor& weight() const noexcept { return weight_; }

    /// @brief Read-only access to the β bias vector (shape [d_model]).
    const Tensor& bias()   const noexcept { return bias_; }

private:
    Tensor weight_;  ///< γ — learned scale, shape [d_model].
    Tensor bias_;    ///< β — learned shift, shape [d_model].

    /// @brief Numerical stability constant (matches PyTorch nn.LayerNorm default).
    static constexpr float kEps = 1e-5f;
};

} // namespace bmgpt
