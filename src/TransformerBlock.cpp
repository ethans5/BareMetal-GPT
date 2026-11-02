/// @file TransformerBlock.cpp
/// @brief Implementation of bmgpt::TransformerBlock (Phase 4 — Transformer complet).

#include "baremetalgpt/TransformerBlock.hpp"

#include <stdexcept>
#include <string>

namespace bmgpt {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

TransformerBlock::TransformerBlock(const WeightLoader& loader,
                                   const std::string&  prefix,
                                   size_t              n_heads)
    : ln_1_(loader, prefix + ".ln_1")
    , attn_(loader, prefix + ".attn", n_heads)
    , ln_2_(loader, prefix + ".ln_2")
    , mlp_ (loader, prefix + ".mlp")
{
    // Cross-check that all sub-layers agree on d_model
    if (attn_.d_model() != ln_1_.d_model()) {
        throw std::runtime_error(
            "TransformerBlock: d_model mismatch between ln_1 (" +
            std::to_string(ln_1_.d_model()) + ") and attn (" +
            std::to_string(attn_.d_model()) + ")");
    }
    if (mlp_.d_model() != ln_1_.d_model()) {
        throw std::runtime_error(
            "TransformerBlock: d_model mismatch between ln_1 (" +
            std::to_string(ln_1_.d_model()) + ") and mlp (" +
            std::to_string(mlp_.d_model()) + ")");
    }
    if (ln_2_.d_model() != ln_1_.d_model()) {
        throw std::runtime_error(
            "TransformerBlock: d_model mismatch between ln_1 (" +
            std::to_string(ln_1_.d_model()) + ") and ln_2 (" +
            std::to_string(ln_2_.d_model()) + ")");
    }
}

// ---------------------------------------------------------------------------
// forward()
// ---------------------------------------------------------------------------

Tensor TransformerBlock::forward(const Tensor& x) const
{
    if (x.ndim() != 2) {
        throw std::invalid_argument(
            "TransformerBlock::forward: expected 2-D input [T, d_model], got ndim=" +
            std::to_string(x.ndim()));
    }
    if (x.size(1) != d_model()) {
        throw std::invalid_argument(
            "TransformerBlock::forward: feature dimension mismatch. "
            "Expected d_model=" + std::to_string(d_model()) +
            ", got " + std::to_string(x.size(1)));
    }

    const size_t T  = x.size(0);
    const size_t dm = d_model();

    // ── Step 1: Attention residual ──────────────────────────────────────────
    // x1 = x + attn(ln_1(x))
    Tensor ln1_out   = ln_1_.forward(x);
    Tensor attn_out  = attn_.forward(ln1_out);

    // Residual addition (element-wise, contiguous [T, dm] tensors)
    Tensor x1({T, dm});
    const float* x_data    = x.data();
    const float* attn_data = attn_out.data();
    float*       x1_data   = x1.data();

    const size_t numel = T * dm;
    for (size_t i = 0; i < numel; ++i) {
        x1_data[i] = x_data[i] + attn_data[i];
    }

    // ── Step 2: MLP residual ────────────────────────────────────────────────
    // x2 = x1 + mlp(ln_2(x1))
    Tensor ln2_out  = ln_2_.forward(x1);
    Tensor mlp_out  = mlp_.forward(ln2_out);

    Tensor x2({T, dm});
    const float* x1d   = x1.data();
    const float* mlp_d = mlp_out.data();
    float*       x2d   = x2.data();

    for (size_t i = 0; i < numel; ++i) {
        x2d[i] = x1d[i] + mlp_d[i];
    }

    return x2;
}

} // namespace bmgpt

