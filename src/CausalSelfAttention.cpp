/// @file CausalSelfAttention.cpp
/// @brief Implementation of bmgpt::CausalSelfAttention (Phase 3 — Layers).

#include "baremetalgpt/CausalSelfAttention.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace bmgpt {

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

CausalSelfAttention::CausalSelfAttention(const WeightLoader& loader,
                                         const std::string& prefix,
                                         size_t n_heads)
    : c_attn_  (loader, prefix + ".c_attn")
    , c_proj_  (loader, prefix + ".c_proj")
    , n_heads_ (n_heads)
    , head_dim_(0)
{
    if (n_heads_ == 0) {
        throw std::runtime_error("CausalSelfAttention: n_heads must be > 0");
    }

    const size_t dm = c_proj_.out_features();
    if (c_proj_.in_features() != dm) {
        throw std::runtime_error(
            "CausalSelfAttention: c_proj must be square [d_model, d_model], got [" +
            std::to_string(c_proj_.out_features()) + ", " +
            std::to_string(c_proj_.in_features()) + "]");
    }
    if (c_attn_.in_features() != dm) {
        throw std::runtime_error(
            "CausalSelfAttention: c_attn in_features (" +
            std::to_string(c_attn_.in_features()) + ") must match d_model (" +
            std::to_string(dm) + ")");
    }
    if (c_attn_.out_features() != 3 * dm) {
        throw std::runtime_error(
            "CausalSelfAttention: c_attn out_features (" +
            std::to_string(c_attn_.out_features()) + ") must be 3 * d_model (" +
            std::to_string(3 * dm) + ")");
    }
    if (dm % n_heads_ != 0) {
        throw std::runtime_error(
            "CausalSelfAttention: d_model (" + std::to_string(dm) +
            ") must be divisible by n_heads (" + std::to_string(n_heads_) + ")");
    }

    head_dim_ = dm / n_heads_;
}

CausalSelfAttention::CausalSelfAttention(Linear c_attn, Linear c_proj, size_t n_heads)
    : c_attn_  (std::move(c_attn))
    , c_proj_  (std::move(c_proj))
    , n_heads_ (n_heads)
    , head_dim_(0)
{
    if (n_heads_ == 0) {
        throw std::runtime_error("CausalSelfAttention: n_heads must be > 0");
    }

    const size_t dm = c_proj_.out_features();
    if (c_proj_.in_features() != dm) {
        throw std::runtime_error(
            "CausalSelfAttention: c_proj must be square [d_model, d_model], got [" +
            std::to_string(c_proj_.out_features()) + ", " +
            std::to_string(c_proj_.in_features()) + "]");
    }
    if (c_attn_.in_features() != dm) {
        throw std::runtime_error(
            "CausalSelfAttention: c_attn in_features (" +
            std::to_string(c_attn_.in_features()) + ") must match d_model (" +
            std::to_string(dm) + ")");
    }
    if (c_attn_.out_features() != 3 * dm) {
        throw std::runtime_error(
            "CausalSelfAttention: c_attn out_features (" +
            std::to_string(c_attn_.out_features()) + ") must be 3 * d_model (" +
            std::to_string(3 * dm) + ")");
    }
    if (dm % n_heads_ != 0) {
        throw std::runtime_error(
            "CausalSelfAttention: d_model (" + std::to_string(dm) +
            ") must be divisible by n_heads (" + std::to_string(n_heads_) + ")");
    }

    head_dim_ = dm / n_heads_;
}

// ---------------------------------------------------------------------------
// forward()
// ---------------------------------------------------------------------------

Tensor CausalSelfAttention::forward(const Tensor& x) const
{
    if (x.ndim() != 2) {
        throw std::invalid_argument(
            "CausalSelfAttention::forward: expected 2-D input [T, d_model], got ndim=" +
            std::to_string(x.ndim()));
    }
    if (x.size(1) != d_model()) {
        throw std::invalid_argument(
            "CausalSelfAttention::forward: feature dimension mismatch. Expected d_model=" +
            std::to_string(d_model()) + ", got " + std::to_string(x.size(1)));
    }

    const size_t T = x.size(0);
    const size_t dm = d_model();
    const size_t D = head_dim_;
    const size_t H = n_heads_;

    if (T == 0) {
        return Tensor({0, dm});
    }

    // 1. QKV projection: [T, d_model] -> [T, 3 * d_model]
    Tensor qkv = c_attn_.forward(x);
    const float* qkv_data = qkv.data();
    const size_t qkv_stride = 3 * dm;

    // 2. Multi-head causal self-attention
    Tensor merged({T, dm});
    float* merged_data = merged.data();

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> scores(T);
    std::vector<float> probs(T);

    for (size_t h = 0; h < H; ++h) {
        const size_t head_offset = h * D;

        for (size_t i = 0; i < T; ++i) {
            // Q pointer for token i, head h
            const float* qi = qkv_data + i * qkv_stride + head_offset;

            // Scaled dot-product with causal history j in [0, i]
            float max_score = -std::numeric_limits<float>::infinity();

            for (size_t j = 0; j <= i; ++j) {
                // K pointer for token j, head h (K starts at offset dm in qkv)
                const float* kj = qkv_data + j * qkv_stride + dm + head_offset;

                float dot = 0.0f;
                for (size_t d = 0; d < D; ++d) {
                    dot += qi[d] * kj[d];
                }
                float s = dot * scale;
                scores[j] = s;
                if (s > max_score) {
                    max_score = s;
                }
            }

            // Numerically stable softmax over [0, i]
            float sum_exp = 0.0f;
            for (size_t j = 0; j <= i; ++j) {
                float p = std::exp(scores[j] - max_score);
                probs[j] = p;
                sum_exp += p;
            }
            const float inv_sum = 1.0f / sum_exp;
            for (size_t j = 0; j <= i; ++j) {
                probs[j] *= inv_sum;
            }

            // Weighted sum over V (V starts at offset 2 * dm in qkv)
            float* out_ih = merged_data + i * dm + head_offset;
            for (size_t d = 0; d < D; ++d) {
                float acc = 0.0f;
                for (size_t j = 0; j <= i; ++j) {
                    const float* vj = qkv_data + j * qkv_stride + 2 * dm + head_offset;
                    acc += probs[j] * vj[d];
                }
                out_ih[d] = acc;
            }
        }
    }

    // 3. Output projection: [T, d_model] -> [T, d_model]
    return c_proj_.forward(merged);
}

} // namespace bmgpt
