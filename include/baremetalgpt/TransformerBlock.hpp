#pragma once

/// @file TransformerBlock.hpp
/// @brief Single Transformer block for BareMetal-GPT (Pre-Norm GPT-2 topology).
///
/// A TransformerBlock implements one decoder layer following the GPT-2 pre-norm
/// architecture:
///
///   x1 = x  + Attention(LayerNorm_1(x))
///   x2 = x1 + MLP(LayerNorm_2(x1))
///
/// Both the LayerNorm and the sub-layer (attention / MLP) are applied to the
/// *same* input — the residual is added *after* the sub-layer output.
/// This is the "pre-norm" (or "pre-LN") variant, as opposed to "post-norm".
///
/// Weight key convention (using WeightLoader, prefix e.g. "h.0"):
///   "<prefix>.ln_1.weight" / "<prefix>.ln_1.bias"
///   "<prefix>.attn.c_attn.weight" / "<prefix>.attn.c_attn.bias"
///   "<prefix>.attn.c_proj.weight" / "<prefix>.attn.c_proj.bias"
///   "<prefix>.ln_2.weight" / "<prefix>.ln_2.bias"
///   "<prefix>.mlp.c_fc.weight"   / "<prefix>.mlp.c_fc.bias"
///   "<prefix>.mlp.c_proj.weight" / "<prefix>.mlp.c_proj.bias"

#include "baremetalgpt/CausalSelfAttention.hpp"
#include "baremetalgpt/FeedForward.hpp"
#include "baremetalgpt/LayerNorm.hpp"
#include "baremetalgpt/Tensor.hpp"
#include "baremetalgpt/WeightLoader.hpp"

#include <string>

namespace bmgpt {

/// @class TransformerBlock
/// @brief Pre-norm Transformer decoder block: LN → Attn (residual) + LN → MLP (residual).
class TransformerBlock {
public:
    // ── Constructor ──────────────────────────────────────────────────────────

    /// @brief Construct a TransformerBlock by loading weights from a WeightLoader.
    /// @param loader   An already-opened WeightLoader pointing at the weight file.
    /// @param prefix   Key prefix for this block, e.g. "h.0" or "h.1".
    ///                 Sub-keys loaded:
    ///                   "<prefix>.ln_1"   → LayerNorm before attention
    ///                   "<prefix>.attn"   → CausalSelfAttention
    ///                   "<prefix>.ln_2"   → LayerNorm before MLP
    ///                   "<prefix>.mlp"    → FeedForward / MLP
    /// @param n_heads  Number of attention heads (must divide d_model).
    /// @throws std::runtime_error on missing keys or dimension mismatch.
    TransformerBlock(const WeightLoader& loader,
                     const std::string&  prefix,
                     size_t              n_heads);

    // ── Forward pass ─────────────────────────────────────────────────────────

    /// @brief Apply one Transformer block to the input hidden states.
    ///
    /// Computes:
    ///   x1 = x  + attn(ln_1(x))
    ///   x2 = x1 + mlp(ln_2(x1))
    ///
    /// @param x  Input tensor of shape [T, d_model].
    /// @return   Output tensor of the same shape [T, d_model].
    /// @throws std::invalid_argument if x is not 2-D or x.size(1) != d_model().
    Tensor forward(const Tensor& x) const;

    // ── Metadata & Sub-layers ────────────────────────────────────────────────

    /// @brief Model embedding dimension (d_model).
    size_t d_model() const noexcept { return ln_1_.d_model(); }

    /// @brief Number of attention heads.
    size_t n_heads() const noexcept { return attn_.n_heads(); }

    /// @brief Read-only access to the first LayerNorm (before attention).
    const LayerNorm& ln_1() const noexcept { return ln_1_; }

    /// @brief Read-only access to the CausalSelfAttention sub-layer.
    const CausalSelfAttention& attn() const noexcept { return attn_; }

    /// @brief Read-only access to the second LayerNorm (before MLP).
    const LayerNorm& ln_2() const noexcept { return ln_2_; }

    /// @brief Read-only access to the FeedForward / MLP sub-layer.
    const FeedForward& mlp() const noexcept { return mlp_; }

private:
    LayerNorm          ln_1_;  ///< Pre-attention layer normalization
    CausalSelfAttention attn_; ///< Multi-head causal self-attention
    LayerNorm          ln_2_;  ///< Pre-MLP layer normalization
    FeedForward        mlp_;   ///< Two-layer feed-forward network
};

} // namespace bmgpt

