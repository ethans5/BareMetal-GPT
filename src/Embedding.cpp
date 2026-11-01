/// @file Embedding.cpp
/// @brief Implementation of bmgpt::Embedding (Phase 4 — Transformer Assembly).

#include "baremetalgpt/Embedding.hpp"

#include <stdexcept>
#include <string>

namespace bmgpt {

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

Embedding::Embedding(const WeightLoader& loader)
    : wte_(loader.load("wte.weight"))
    , wpe_(loader.load("wpe.weight"))
{
    // WeightLoader::load() already validates that the tensors exist and
    // are float32. We only need to verify their ranks and consistency here.
    if (wte_.ndim() != 2) {
        throw std::runtime_error(
            "Embedding: wte.weight must be 2-D [vocab_size, d_model], got ndim=" +
            std::to_string(wte_.ndim()));
    }
    if (wpe_.ndim() != 2) {
        throw std::runtime_error(
            "Embedding: wpe.weight must be 2-D [context_length, d_model], got ndim=" +
            std::to_string(wpe_.ndim()));
    }
    if (wte_.size(1) != wpe_.size(1)) {
        throw std::runtime_error(
            "Embedding: d_model mismatch between wte (" +
            std::to_string(wte_.size(1)) + ") and wpe (" +
            std::to_string(wpe_.size(1)) + ")");
    }
}

Embedding::Embedding(Tensor wte, Tensor wpe)
    : wte_(std::move(wte))
    , wpe_(std::move(wpe))
{
    if (wte_.ndim() != 2) {
        throw std::invalid_argument(
            "Embedding: wte must be 2-D [vocab_size, d_model], got ndim=" +
            std::to_string(wte_.ndim()));
    }
    if (wpe_.ndim() != 2) {
        throw std::invalid_argument(
            "Embedding: wpe must be 2-D [context_length, d_model], got ndim=" +
            std::to_string(wpe_.ndim()));
    }
    if (wte_.size(1) != wpe_.size(1)) {
        throw std::invalid_argument(
            "Embedding: d_model mismatch between wte (" +
            std::to_string(wte_.size(1)) + ") and wpe (" +
            std::to_string(wpe_.size(1)) + ")");
    }
}

// ---------------------------------------------------------------------------
// forward()
// ---------------------------------------------------------------------------

Tensor Embedding::forward(const std::vector<int32_t>& input_ids) const
{
    // ── Guard: empty sequence ──────────────────────────────────────────────
    if (input_ids.empty()) {
        throw std::invalid_argument(
            "Embedding::forward: input_ids must not be empty");
    }

    const size_t T       = input_ids.size();
    const size_t V       = vocab_size();
    const size_t C       = context_length();
    const size_t D       = d_model();

    // ── Guard: sequence too long ───────────────────────────────────────────
    if (T > C) {
        throw std::length_error(
            "Embedding::forward: sequence length " + std::to_string(T) +
            " exceeds context_length " + std::to_string(C));
    }

    // ── Guard: token IDs in range ──────────────────────────────────────────
    for (size_t i = 0; i < T; ++i) {
        const int32_t id = input_ids[i];
        if (id < 0) {
            throw std::out_of_range(
                "Embedding::forward: negative token ID " + std::to_string(id) +
                " at position " + std::to_string(i));
        }
        if (static_cast<size_t>(id) >= V) {
            throw std::out_of_range(
                "Embedding::forward: token ID " + std::to_string(id) +
                " at position " + std::to_string(i) +
                " is out of range [0, " + std::to_string(V) + ")");
        }
    }

    // ── Allocate output tensor [T, D] ──────────────────────────────────────
    Tensor out({T, D});
    float* dst = out.data();

    // Raw pointers for fast row-copy arithmetic
    const float* wte_data = wte_.data();
    const float* wpe_data = wpe_.data();

    // ── Elementwise sum: out[i, :] = wte[token_id, :] + wpe[i, :] ─────────
    for (size_t i = 0; i < T; ++i) {
        const size_t tok_row = static_cast<size_t>(input_ids[i]);
        const float* tok_src = wte_data + tok_row * D;  // wte row for this token
        const float* pos_src = wpe_data + i       * D;  // wpe row for position i
        float*       out_row = dst        + i       * D;

        for (size_t d = 0; d < D; ++d) {
            out_row[d] = tok_src[d] + pos_src[d];
        }
    }

    return out;
}

} // namespace bmgpt
