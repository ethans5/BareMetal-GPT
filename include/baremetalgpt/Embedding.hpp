#pragma once

/// @file Embedding.hpp
/// @brief Token and positional embedding table for GPT-2.
///
/// GPT-2 uses two learned embedding tables:
///   - wte (word token embedding): maps each token ID to a d_model-dimensional vector.
///   - wpe (word position embedding): maps each position index to a d_model-dimensional vector.
///
/// The initial hidden state fed into the first Transformer block is the sum of both:
///
///   h0[i, :] = wte[input_ids[i], :] + wpe[i, :]
///
/// Weight keys (matching export_weights.py and WeightLoader convention):
///   - "wte.weight"  shape [vocab_size, d_model]
///   - "wpe.weight"  shape [context_length, d_model]
///
/// The wte() accessor is intentionally public so that Model can reuse the
/// wte weight matrix for the final vocabulary projection (weight tying):
///   logits = final_hidden @ wte.weight^T

#include "Tensor.hpp"
#include "WeightLoader.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace bmgpt {

/// @class Embedding
/// @brief Sums token embeddings (wte) and learned positional embeddings (wpe).
class Embedding {
public:
    // ── Constructors ──────────────────────────────────────────────────────────

    /// @brief Construct Embedding by loading weights from a WeightLoader.
    /// @param loader  Opened WeightLoader pointing at a .safetensors file.
    /// @throws std::runtime_error if "wte.weight" or "wpe.weight" are missing
    ///         or have unexpected shapes.
    explicit Embedding(const WeightLoader& loader);

    /// @brief Construct Embedding by directly injecting weight tensors.
    /// @param wte  Token embedding matrix, shape [vocab_size, d_model].
    /// @param wpe  Position embedding matrix, shape [context_length, d_model].
    /// @throws std::invalid_argument if either tensor is not 2-D, or if their
    ///         second dimensions (d_model) do not match.
    Embedding(Tensor wte, Tensor wpe);

    // ── Forward pass ─────────────────────────────────────────────────────────

    /// @brief Compute h0 = wte[input_ids] + wpe[0..T-1].
    /// @param input_ids  Sequence of token IDs, length T.
    /// @return           Tensor of shape [T, d_model].
    /// @throws std::invalid_argument  if input_ids is empty.
    /// @throws std::out_of_range      if any token ID is negative or >= vocab_size.
    /// @throws std::length_error      if T > context_length.
    Tensor forward(const std::vector<int32_t>& input_ids) const;

    // ── Metadata & weight accessors ───────────────────────────────────────────

    /// @brief Vocabulary size (number of rows in wte).
    size_t vocab_size()     const noexcept { return wte_.size(0); }

    /// @brief Maximum sequence length (number of rows in wpe).
    size_t context_length() const noexcept { return wpe_.size(0); }

    /// @brief Embedding dimension (d_model).
    size_t d_model()        const noexcept { return wte_.size(1); }

    /// @brief Read-only access to the token embedding matrix [vocab_size, d_model].
    /// Used by Model for weight tying (logits = final_hidden @ wte().transpose()).
    const Tensor& wte() const noexcept { return wte_; }

    /// @brief Read-only access to the positional embedding matrix [context_length, d_model].
    const Tensor& wpe() const noexcept { return wpe_; }

private:
    Tensor wte_;  ///< [vocab_size, d_model]
    Tensor wpe_;  ///< [context_length, d_model]
};

} // namespace bmgpt
