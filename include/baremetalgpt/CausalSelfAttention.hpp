#pragma once

/// @file CausalSelfAttention.hpp
/// @brief Multi-head Causal Self-Attention layer for BareMetal-GPT.
///
/// Implements standard scaled dot-product causal self-attention as used in GPT-2:
///
///   1. QKV projection:       [T, d_model] -> [T, 3 * d_model] via c_attn
///   2. Split heads:          Q, K, V -> [n_heads, T, head_dim]
///   3. Scaled dot-product:   scores = (Q @ K^T) / sqrt(head_dim)
///   4. Causal masking:       scores[i, j] = -inf for j > i
///   5. Softmax:              attn_probs = softmax(scores, dim=-1)
///   6. Weighted sum:         out_heads = attn_probs @ V -> [n_heads, T, head_dim]
///   7. Merge heads:          merged = concat(out_heads) -> [T, d_model]
///   8. Output projection:    y = merged @ W_proj^T + b_proj via c_proj
///
/// Note: The residual addition (x + y) is NOT performed here,
/// but in TransformerBlock, exactly following GPT-2 architecture.

#include "Linear.hpp"
#include "Tensor.hpp"
#include "WeightLoader.hpp"

#include <string>

namespace bmgpt {

/// @class CausalSelfAttention
/// @brief Multi-head causal self-attention layer.
class CausalSelfAttention {
public:
    // ── Constructors ────────────────────────────────────────────────

    /// @brief Construct by loading weights from WeightLoader.
    /// @param loader   Opened WeightLoader.
    /// @param prefix   Prefix path, e.g. "h.0.attn".
    ///                 Sub-keys: "<prefix>.c_attn" [3*d_model, d_model]
    ///                           "<prefix>.c_proj" [d_model, d_model]
    /// @param n_heads  Number of attention heads (must divide d_model).
    /// @throws std::runtime_error on missing keys or dimension mismatch.
    CausalSelfAttention(const WeightLoader& loader, const std::string& prefix, size_t n_heads);

    /// @brief Construct directly from two Linear layers and head count.
    /// @param c_attn   Fused QKV projection layer.
    /// @param c_proj   Output projection layer.
    /// @param n_heads  Number of attention heads.
    /// @throws std::runtime_error on dimension mismatch.
    CausalSelfAttention(Linear c_attn, Linear c_proj, size_t n_heads);

    // ── Forward pass ────────────────────────────────────────────────

    /// @brief Compute causal multi-head self-attention.
    /// @param x  Input tensor of shape [T, d_model].
    /// @return   Output tensor of shape [T, d_model].
    /// @throws std::invalid_argument if x is not 2-D or x.size(1) != d_model().
    Tensor forward(const Tensor& x) const;

    // ── Metadata & Sublayers ────────────────────────────────────────

    /// @brief Number of attention heads.
    size_t n_heads() const noexcept { return n_heads_; }

    /// @brief Model embedding dimension (d_model).
    size_t d_model() const noexcept { return c_proj_.out_features(); }

    /// @brief Dimension of each attention head (d_model / n_heads).
    size_t head_dim() const noexcept { return head_dim_; }

    /// @brief Access fused QKV projection layer (c_attn).
    const Linear& c_attn() const noexcept { return c_attn_; }

    /// @brief Access output projection layer (c_proj).
    const Linear& c_proj() const noexcept { return c_proj_; }

private:
    Linear c_attn_;    ///< Fused QKV projection [d_model -> 3 * d_model]
    Linear c_proj_;    ///< Output projection [d_model -> d_model]
    size_t n_heads_;   ///< Number of attention heads
    size_t head_dim_;  ///< Head dimension = d_model / n_heads
};

} // namespace bmgpt
