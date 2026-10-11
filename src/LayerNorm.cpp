/// @file LayerNorm.cpp
/// @brief Implementation of bmgpt::LayerNorm (Phase 3 — Layers).
///
/// Algorithm (row-wise, matching PyTorch nn.LayerNorm):
///   For each token position t in [0, T):
///     1. Compute mean   μ  = (1/D) Σ x[t, j]
///     2. Compute variance σ² = (1/D) Σ (x[t, j] - μ)²
///     3. Normalize       x̂[t, j] = (x[t, j] - μ) / √(σ² + ε)
///     4. Affine scale    y[t, j]  = γ[j] · x̂[t, j] + β[j]
///
/// The two-pass approach (separate loops for mean and variance) avoids
/// catastrophic cancellation compared to the single-pass variance formula.

#include "baremetalgpt/LayerNorm.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace bmgpt {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

LayerNorm::LayerNorm(const WeightLoader& loader, const std::string& prefix)
    : weight_(loader.load(prefix + ".weight"))
    , bias_  (loader.load(prefix + ".bias"))
{
    if (weight_.ndim() != 1) {
        throw std::runtime_error(
            "LayerNorm: weight tensor for \"" + prefix + "\" must be 1-D, "
            "got ndim=" + std::to_string(weight_.ndim()));
    }
    if (bias_.ndim() != 1) {
        throw std::runtime_error(
            "LayerNorm: bias tensor for \"" + prefix + "\" must be 1-D, "
            "got ndim=" + std::to_string(bias_.ndim()));
    }
    if (weight_.numel() != bias_.numel()) {
        throw std::runtime_error(
            "LayerNorm: weight/bias size mismatch for \"" + prefix + "\": "
            + std::to_string(weight_.numel()) + " vs "
            + std::to_string(bias_.numel()));
    }
}

// ---------------------------------------------------------------------------
// forward()
// ---------------------------------------------------------------------------

Tensor LayerNorm::forward(const Tensor& x) const
{
    // ── Input validation ────────────────────────────────────────────
    if (x.ndim() != 2) {
        throw std::invalid_argument(
            "LayerNorm::forward: expected 2-D input [T, d_model], "
            "got ndim=" + std::to_string(x.ndim()));
    }
    const size_t T = x.size(0);
    const size_t D = x.size(1);

    if (D != d_model()) {
        throw std::invalid_argument(
            "LayerNorm::forward: input last dim (" + std::to_string(D) +
            ") does not match d_model (" + std::to_string(d_model()) + ")");
    }

    // ── Allocate output ─────────────────────────────────────────────
    Tensor out({T, D});

    const float* x_ptr   = x.data();
    const float* w_ptr   = weight_.data();
    const float* b_ptr   = bias_.data();
    float*       out_ptr = out.data();

    // ── Row-wise normalization ──────────────────────────────────────
    for (size_t t = 0; t < T; ++t) {
        const float* row     = x_ptr   + t * D;
        float*       out_row = out_ptr + t * D;

        // Pass 1 — mean
        float mean = 0.0f;
        for (size_t j = 0; j < D; ++j) {
            mean += row[j];
        }
        mean /= static_cast<float>(D);

        // Pass 2 — variance (biased, matching PyTorch LayerNorm)
        float var = 0.0f;
        for (size_t j = 0; j < D; ++j) {
            float diff = row[j] - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(D);

        // Reciprocal standard deviation (fused with ε for one sqrt call per row)
        float inv_std = 1.0f / std::sqrt(var + kEps);

        // Pass 3 — normalize + affine
        for (size_t j = 0; j < D; ++j) {
            out_row[j] = w_ptr[j] * ((row[j] - mean) * inv_std) + b_ptr[j];
        }
    }

    return out;
}

} // namespace bmgpt
