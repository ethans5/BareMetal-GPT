#pragma once

/// @file FeedForward.hpp
/// @brief Multi-Layer Perceptron (MLP) / Feed-Forward Network for GPT-2.
///
/// Implements the two-layer feed-forward network with GELU activation:
///
///   FFN(x) = Linear_proj(GELU(Linear_fc(x)))
///
/// where:
///   - Linear_fc: projects from d_model to d_ff (usually 4 * d_model).
///     Corresponds to weight key "<prefix>.c_fc.{weight,bias}".
///   - GELU: element-wise tanh approximation variant (gelu_new).
///   - Linear_proj: projects from d_ff back to d_model.
///     Corresponds to weight key "<prefix>.c_proj.{weight,bias}".
///
/// Dimensions:
///   - Input  x:      [T, d_model]
///   - c_fc out:      [T, d_ff]
///   - GELU out:      [T, d_ff]
///   - Output y:      [T, d_model]
///
/// Note: The residual addition (x + y) is NOT performed inside this class,
/// but by the enclosing TransformerBlock, exactly following GPT-2 architecture.

#include "Linear.hpp"
#include "activations.hpp"
#include "WeightLoader.hpp"

#include <string>

namespace bmgpt {

/// @class FeedForward
/// @brief Two-layer MLP with GELU activation: Linear(c_fc) -> GELU -> Linear(c_proj).
class FeedForward {
public:
    // ── Constructors ────────────────────────────────────────────────

    /// @brief Construct FeedForward by loading weights from WeightLoader.
    /// @param loader   Opened WeightLoader.
    /// @param prefix   Prefix path, e.g. "h.0.mlp".
    ///                 Sub-keys loaded: "<prefix>.c_fc" and "<prefix>.c_proj".
    /// @throws std::runtime_error on missing keys or dimension mismatch between layers.
    FeedForward(const WeightLoader& loader, const std::string& prefix);

    /// @brief Construct FeedForward directly from two Linear layers.
    /// @param c_fc    First projection (d_model -> d_ff).
    /// @param c_proj  Second projection (d_ff -> d_model).
    /// @throws std::runtime_error if inner dimensions do not match.
    FeedForward(Linear c_fc, Linear c_proj);

    // ── Forward pass ────────────────────────────────────────────────

    /// @brief Compute y = c_proj(GELU(c_fc(x))).
    /// @param x  Input tensor of shape [T, d_model].
    /// @return   Output tensor of shape [T, d_model].
    /// @throws std::invalid_argument if x is not 2-D or x.size(1) != d_model().
    Tensor forward(const Tensor& x) const;

    // ── Metadata & Sublayers ────────────────────────────────────────

    /// @brief Input/output model dimension (d_model).
    size_t d_model() const noexcept { return c_fc_.in_features(); }

    /// @brief Hidden intermediate dimension (d_ff).
    size_t d_ff() const noexcept { return c_fc_.out_features(); }

    /// @brief Access first linear layer (c_fc).
    const Linear& c_fc() const noexcept { return c_fc_; }

    /// @brief Access second linear layer (c_proj).
    const Linear& c_proj() const noexcept { return c_proj_; }

private:
    Linear c_fc_;    ///< First projection [d_model -> d_ff]
    Linear c_proj_;  ///< Second projection [d_ff -> d_model]
};

/// Friendly alias matching HuggingFace GPT2MLP
using MLP = FeedForward;

} // namespace bmgpt
