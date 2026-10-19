/// @file FeedForward.cpp
/// @brief Implementation of bmgpt::FeedForward (Phase 3 — Layers).

#include "baremetalgpt/FeedForward.hpp"

#include <stdexcept>
#include <string>

namespace bmgpt {

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

FeedForward::FeedForward(const WeightLoader& loader, const std::string& prefix)
    : c_fc_  (loader, prefix + ".c_fc")
    , c_proj_(loader, prefix + ".c_proj")
{
    if (c_fc_.out_features() != c_proj_.in_features()) {
        throw std::runtime_error(
            "FeedForward: dimension mismatch between c_fc out (" +
            std::to_string(c_fc_.out_features()) + ") and c_proj in (" +
            std::to_string(c_proj_.in_features()) + ") for \"" + prefix + "\"");
    }
    if (c_fc_.in_features() != c_proj_.out_features()) {
        throw std::runtime_error(
            "FeedForward: dimension mismatch between c_fc in (" +
            std::to_string(c_fc_.in_features()) + ") and c_proj out (" +
            std::to_string(c_proj_.out_features()) + ") for \"" + prefix + "\"");
    }
}

FeedForward::FeedForward(Linear c_fc, Linear c_proj)
    : c_fc_  (std::move(c_fc))
    , c_proj_(std::move(c_proj))
{
    if (c_fc_.out_features() != c_proj_.in_features()) {
        throw std::runtime_error(
            "FeedForward: dimension mismatch between c_fc out (" +
            std::to_string(c_fc_.out_features()) + ") and c_proj in (" +
            std::to_string(c_proj_.in_features()) + ")");
    }
    if (c_fc_.in_features() != c_proj_.out_features()) {
        throw std::runtime_error(
            "FeedForward: dimension mismatch between c_fc in (" +
            std::to_string(c_fc_.in_features()) + ") and c_proj out (" +
            std::to_string(c_proj_.out_features()) + ")");
    }
}

// ---------------------------------------------------------------------------
// forward()
// ---------------------------------------------------------------------------

Tensor FeedForward::forward(const Tensor& x) const
{
    if (x.ndim() != 2) {
        throw std::invalid_argument(
            "FeedForward::forward: expected 2-D input [T, d_model], got ndim=" +
            std::to_string(x.ndim()));
    }
    if (x.size(1) != d_model()) {
        throw std::invalid_argument(
            "FeedForward::forward: feature dimension mismatch. Expected d_model=" +
            std::to_string(d_model()) + ", got " + std::to_string(x.size(1)));
    }

    // 1. Expand projection: [T, d_model] -> [T, d_ff]
    Tensor h = c_fc_.forward(x);

    // 2. Element-wise GELU (tanh approximation)
    Tensor act = gelu(h);

    // 3. Contract projection: [T, d_ff] -> [T, d_model]
    return c_proj_.forward(act);
}

} // namespace bmgpt
