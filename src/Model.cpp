/// @file Model.cpp
/// @brief Implementation of bmgpt::Model — full GPT-2 forward pass (Phase 4).

#include "baremetalgpt/Model.hpp"
#include "baremetalgpt/matmul.hpp"

#include <stdexcept>
#include <string>

namespace bmgpt {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

Model::Model(const WeightLoader& loader, const ModelConfig& config)
    : config_   (config)
    , embedding_(loader)
    , ln_f_     (loader, "ln_f")
    , wte_T_    ()
{
    // Validate that embedding dimensions match the declared config
    if (embedding_.d_model() != config_.d_model) {
        throw std::runtime_error(
            "Model: d_model mismatch — config says " +
            std::to_string(config_.d_model) +
            " but wte has d_model=" +
            std::to_string(embedding_.d_model()));
    }
    if (embedding_.vocab_size() != config_.vocab_size) {
        throw std::runtime_error(
            "Model: vocab_size mismatch — config says " +
            std::to_string(config_.vocab_size) +
            " but wte has vocab_size=" +
            std::to_string(embedding_.vocab_size()));
    }
    if (embedding_.context_length() != config_.context_length) {
        throw std::runtime_error(
            "Model: context_length mismatch — config says " +
            std::to_string(config_.context_length) +
            " but wpe has context_length=" +
            std::to_string(embedding_.context_length()));
    }

    // Build the N Transformer blocks
    blocks_.reserve(config_.n_layers);
    for (size_t i = 0; i < config_.n_layers; ++i) {
        blocks_.emplace_back(loader, "h." + std::to_string(i), config_.n_heads);
    }

    // Pre-compute wte^T = wte.weight^T for cache-friendly weight-tying matmul.
    // wte.weight shape: [vocab_size, d_model]
    // wte_T_     shape: [d_model,    vocab_size]
    wte_T_ = embedding_.wte().transpose(0, 1);
}

// ---------------------------------------------------------------------------
// forward()
// ---------------------------------------------------------------------------

Tensor Model::forward(const std::vector<int32_t>& input_ids) const
{
    // 1. Token + positional embeddings: [T, d_model]
    Tensor h = embedding_.forward(input_ids);

    // 2. Pass through each Transformer block
    for (const TransformerBlock& block : blocks_) {
        h = block.forward(h);
    }

    // 3. Final layer normalization
    h = ln_f_.forward(h);

    // 4. Vocabulary projection via weight tying:
    //    logits = h @ wte^T   shape: [T, vocab_size]
    return matmul(h, wte_T_);
}

} // namespace bmgpt

