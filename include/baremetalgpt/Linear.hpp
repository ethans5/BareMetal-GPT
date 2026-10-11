#pragma once

/// @file Linear.hpp
/// @brief Fully-connected linear layer for the BareMetal-GPT inference engine.
///
/// Implements the standard affine transformation used in every projection
/// inside GPT-2:
///
///   y = x · Wᵀ + b
///
/// where:
///   - x has shape [T, in_features]
///   - W has shape [out_features, in_features]  (stored as [out, in], standard)
///   - b has shape [out_features]
///   - y has shape [T, out_features]
///
/// This matches PyTorch's `nn.Linear(in_features, out_features)`.
/// Note: the HuggingFace GPT-2 Conv1D weights are **pre-transposed** at
/// export time (scripts/export_weights.py) so they arrive here already in
/// [out, in] order — no special casing is needed in this class.
///
/// Internal computation:
///   1. Compute  x @ Wᵀ  via bmgpt::matmul (cache-friendly i-k-j loop).
///   2. Broadcast-add bias b to every row.
///
/// Usage:
/// @code
///   bmgpt::WeightLoader loader("weights/micro_gpt2_weights.safetensors");
///   bmgpt::Linear qkv_proj(loader, "h.0.attn.c_attn");  // weight + bias
///   Tensor out = qkv_proj.forward(ln1_out);              // [T,64] → [T,192]
/// @endcode

#include "Tensor.hpp"
#include "WeightLoader.hpp"

#include <string>

namespace bmgpt {

/// @class Linear
/// @brief Affine projection layer: y = x · Wᵀ + b.
///
/// Weights are loaded once at construction. The forward pass is stateless
/// (no mutable state after construction).
class Linear {
public:
    // ── Constructor ─────────────────────────────────────────────────

    /// @brief Load weight W and bias b from a .safetensors file.
    /// @param loader   An already-opened WeightLoader.
    /// @param prefix   Key prefix, e.g. "h.0.attn.c_attn".
    ///                 Looks up "<prefix>.weight" [out, in] and "<prefix>.bias" [out].
    /// @throws std::runtime_error if either key is missing, if weight is not 2-D,
    ///         or if bias size does not match weight's out dimension.
    Linear(const WeightLoader& loader, const std::string& prefix);

    // ── Forward pass ─────────────────────────────────────────────────

    /// @brief Compute y = x · Wᵀ + b.
    /// @param x  Input tensor of shape [T, in_features].
    /// @return   Output tensor of shape [T, out_features].
    /// @throws std::invalid_argument if x is not 2-D, or if x.size(1) !=
    ///         in_features().
    Tensor forward(const Tensor& x) const;

    // ── Metadata ─────────────────────────────────────────────────────

    /// @brief Number of input features (= weight.size(1)).
    size_t in_features()  const noexcept { return weight_.size(1); }

    /// @brief Number of output features (= weight.size(0) = bias.numel()).
    size_t out_features() const noexcept { return weight_.size(0); }

    /// @brief Read-only access to the weight matrix W, shape [out, in].
    const Tensor& weight() const noexcept { return weight_; }

    /// @brief Read-only access to the bias vector b, shape [out].
    const Tensor& bias()   const noexcept { return bias_; }

private:
    Tensor weight_;  ///< W — shape [out_features, in_features].
    Tensor bias_;    ///< b — shape [out_features].
};

} // namespace bmgpt
