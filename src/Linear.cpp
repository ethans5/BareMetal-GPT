/// @file Linear.cpp
/// @brief Implementation of bmgpt::Linear (Phase 3 — Layers).
///
/// Forward pass algorithm:
///   1. Transpose W from [out, in] to [in, out]  (matmul expects [K, N]).
///   2. y_no_bias = matmul(x, W^T)               (shape [T, out]).
///   3. Broadcast-add bias: y[t, j] += b[j]       for all t, j.
///
/// Why transpose W before calling matmul()?
///   bmgpt::matmul(a, b) computes a @ b, requiring b to be [K, N].
///   W is stored as [out, in] = [N, K], so we need Wᵀ = [in, out] = [K, N].
///   Alternatively we could compute matmul(x, W^T) directly by calling
///   W.transpose(0,1) which is O(numel) but physically reorders the data
///   for cache-friendly access in the inner loop.

#include "baremetalgpt/Linear.hpp"
#include "baremetalgpt/matmul.hpp"

#include <stdexcept>
#include <string>

namespace bmgpt {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

Linear::Linear(const WeightLoader& loader, const std::string& prefix)
    : weight_(loader.load(prefix + ".weight"))
    , bias_  (loader.load(prefix + ".bias"))
{
    if (weight_.ndim() != 2) {
        throw std::runtime_error(
            "Linear: weight for \"" + prefix + "\" must be 2-D [out, in], "
            "got ndim=" + std::to_string(weight_.ndim()));
    }
    if (bias_.ndim() != 1) {
        throw std::runtime_error(
            "Linear: bias for \"" + prefix + "\" must be 1-D [out], "
            "got ndim=" + std::to_string(bias_.ndim()));
    }
    if (bias_.numel() != weight_.size(0)) {
        throw std::runtime_error(
            "Linear: bias size (" + std::to_string(bias_.numel()) +
            ") does not match weight out_features (" +
            std::to_string(weight_.size(0)) + ") for \"" + prefix + "\"");
    }
}

// ---------------------------------------------------------------------------
// forward()
// ---------------------------------------------------------------------------

Tensor Linear::forward(const Tensor& x) const
{
    // ── Input validation ────────────────────────────────────────────
    if (x.ndim() != 2) {
        throw std::invalid_argument(
            "Linear::forward: expected 2-D input [T, in_features], "
            "got ndim=" + std::to_string(x.ndim()));
    }
    if (x.size(1) != in_features()) {
        throw std::invalid_argument(
            "Linear::forward: input size(1)=" + std::to_string(x.size(1)) +
            " does not match in_features=" + std::to_string(in_features()));
    }

    // ── Step 1: Wᵀ — [out, in] → [in, out] ─────────────────────────
    // transpose(0,1) physically reorders memory for cache-friendly matmul.
    Tensor wt = weight_.transpose(0, 1);  // [in_features, out_features]

    // ── Step 2: x @ Wᵀ ──────────────────────────────────────────────
    // matmul([T, in], [in, out]) → [T, out]
    Tensor out = matmul(x, wt);

    // ── Step 3: broadcast-add bias ───────────────────────────────────
    const size_t T   = out.size(0);
    const size_t N   = out.size(1);  // = out_features()
    float*       dst = out.data();
    const float* b   = bias_.data();

    for (size_t t = 0; t < T; ++t) {
        float* row = dst + t * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] += b[j];
        }
    }

    return out;
}

} // namespace bmgpt
