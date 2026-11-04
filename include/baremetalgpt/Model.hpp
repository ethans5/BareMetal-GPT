#pragma once

/// @file Model.hpp
/// @brief Full GPT-2 Model: Tokens → Logits (Phase 4).
///
/// Pipeline:
///   1. h = Embedding(input_ids)                   [T, d_model]
///   2. For each block i in [0, n_layers):
///      h = TransformerBlock_i(h)                  [T, d_model]
///   3. h = LayerNorm_f(h)                         [T, d_model]
///   4. logits = h @ wte^T                         [T, vocab_size]  ← weight tying
///
/// Weight tying: the vocabulary projection matrix is the *transpose* of the
/// token embedding matrix (wte.weight), following the original GPT-2 design.
/// wte^T is pre-computed at construction time for cache-friendly matmul.
///
/// Weight key convention (WeightLoader):
///   "wte.weight"          [vocab_size, d_model]
///   "wpe.weight"          [context_length, d_model]
///   "h.<i>.ln_1.weight"   [d_model]  (for i in [0, n_layers))
///   "h.<i>.ln_1.bias"     [d_model]
///   "h.<i>.attn.*"        (CausalSelfAttention sub-keys)
///   "h.<i>.ln_2.weight"   [d_model]
///   "h.<i>.ln_2.bias"     [d_model]
///   "h.<i>.mlp.*"         (FeedForward sub-keys)
///   "ln_f.weight"         [d_model]
///   "ln_f.bias"           [d_model]

#include "baremetalgpt/Embedding.hpp"
#include "baremetalgpt/LayerNorm.hpp"
#include "baremetalgpt/Tensor.hpp"
#include "baremetalgpt/TransformerBlock.hpp"
#include "baremetalgpt/WeightLoader.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bmgpt {

// ---------------------------------------------------------------------------
// ModelConfig
// ---------------------------------------------------------------------------

/// @struct ModelConfig
/// @brief Hyperparameters that describe a GPT-2 variant.
struct ModelConfig {
    size_t n_layers;        ///< Number of Transformer blocks.
    size_t n_heads;         ///< Number of attention heads per block.
    size_t d_model;         ///< Embedding / hidden dimension.
    size_t vocab_size;      ///< Vocabulary size (rows in wte).
    size_t context_length;  ///< Maximum sequence length (rows in wpe).
    size_t d_ff;            ///< Feed-forward hidden dimension (usually 4 * d_model).
    float  eps;             ///< LayerNorm epsilon (default 1e-5).
};

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// @class Model
/// @brief Full GPT-2 inference model: Embedding → N × TransformerBlock → LN → Logits.
class Model {
public:
    // ── Constructors ─────────────────────────────────────────────────────────

    /// @brief Construct a Model by loading all weights from a WeightLoader.
    ///
    /// The number of Transformer blocks is inferred from @p config.n_layers.
    /// wte^T is pre-computed at construction for efficient weight-tying matmul.
    ///
    /// @param loader  Opened WeightLoader pointing at a .safetensors file.
    /// @param config  Hyperparameters (n_layers, n_heads, d_model, …).
    /// @throws std::runtime_error on missing keys or dimension mismatches.
    Model(const WeightLoader& loader, const ModelConfig& config);

    // ── Forward pass ─────────────────────────────────────────────────────────

    /// @brief Run a full GPT-2 forward pass and return unnormalized logits.
    ///
    /// @param input_ids  Sequence of token IDs, length T (1 ≤ T ≤ context_length).
    /// @return           Tensor of shape [T, vocab_size] — raw (pre-softmax) logits.
    /// @throws std::invalid_argument  if input_ids is empty.
    /// @throws std::out_of_range      if any token ID is out of [0, vocab_size).
    /// @throws std::length_error      if T > context_length.
    Tensor forward(const std::vector<int32_t>& input_ids) const;

    // ── Metadata ──────────────────────────────────────────────────────────────

    /// @brief Model configuration (hyperparameters).
    const ModelConfig& config() const noexcept { return config_; }

    /// @brief Number of Transformer blocks.
    size_t n_layers()       const noexcept { return config_.n_layers; }

    /// @brief Number of attention heads.
    size_t n_heads()        const noexcept { return config_.n_heads; }

    /// @brief Hidden embedding dimension.
    size_t d_model()        const noexcept { return config_.d_model; }

    /// @brief Vocabulary size.
    size_t vocab_size()     const noexcept { return config_.vocab_size; }

    /// @brief Maximum context length.
    size_t context_length() const noexcept { return config_.context_length; }

    // ── Sub-layer accessors (useful for unit tests) ───────────────────────────

    /// @brief Token + position embedding layer.
    const Embedding& embedding() const noexcept { return embedding_; }

    /// @brief The i-th Transformer block (0-indexed).
    const TransformerBlock& block(size_t i) const { return blocks_.at(i); }

    /// @brief Final layer normalization (applied before logit projection).
    const LayerNorm& ln_f() const noexcept { return ln_f_; }

    /// @brief Pre-computed wte^T used for the vocabulary projection (weight tying).
    /// Shape: [d_model, vocab_size].
    const Tensor& wte_T() const noexcept { return wte_T_; }

private:
    ModelConfig                 config_;    ///< Hyperparameters
    Embedding                   embedding_; ///< wte + wpe
    std::vector<TransformerBlock> blocks_;  ///< N Transformer decoder blocks
    LayerNorm                   ln_f_;      ///< Final layer normalization
    Tensor                      wte_T_;     ///< wte transposed [d_model, vocab_size]
};

} // namespace bmgpt

