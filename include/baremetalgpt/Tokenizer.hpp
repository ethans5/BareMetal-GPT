#pragma once

/// @file Tokenizer.hpp
/// @brief Byte-Level Byte-Pair Encoding (BPE) Tokenizer for GPT-2.
///
/// Implements the exact tokenizer used by GPT-2 (Radford et al., 2019):
///   1. Pre-tokenization: regex splitting on words, contractions, numbers, punctuation, spaces.
///   2. Byte-to-Unicode mapping: invertible mapping from all 256 byte values to printable Unicode codepoints.
///   3. Byte-Pair Encoding (BPE): greedy pair merging driven by merges.txt ranks.
///   4. Vocabulary lookup: mapping merged symbols to token IDs via vocab.json.
///
/// Properties:
///   - 100% lossless: decode(encode(text)) == text for any UTF-8 input.
///   - Zero unknown tokens: any arbitrary byte sequence can be represented.
///   - Standard vocab_size = 50257, eos_token_id = 50256 (<|endoftext|>).

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bmgpt {

/// @struct PairHash
/// @brief Hash function for std::pair<std::string, std::string> used in BPE merge ranks.
struct PairHash {
    std::size_t operator()(const std::pair<std::string, std::string>& p) const noexcept {
        // Standard combination hash
        std::size_t h1 = std::hash<std::string>{}(p.first);
        std::size_t h2 = std::hash<std::string>{}(p.second);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

/// @class Tokenizer
/// @brief Full GPT-2 Byte-Level BPE Tokenizer.
class Tokenizer {
public:
    // ── Constructors ─────────────────────────────────────────────────────────

    /// @brief Construct Tokenizer by loading vocab.json and merges.txt files.
    /// @param vocab_file   Path to vocab.json (token string -> id).
    /// @param merges_file  Path to merges.txt (merge pairs in priority order).
    /// @throws std::runtime_error on missing files or malformed content.
    Tokenizer(const std::string& vocab_file, const std::string& merges_file);

    // ── Tokenization & Detokenization ─────────────────────────────────────────

    /// @brief Encode a raw UTF-8 string into a sequence of token IDs.
    /// @param text  Input text to tokenize.
    /// @return      Vector of token IDs (values in [0, vocab_size)).
    std::vector<int32_t> encode(const std::string& text) const;

    /// @brief Decode a sequence of token IDs back into a UTF-8 string.
    /// @param tokens  Vector of token IDs.
    /// @return        Reconstructed UTF-8 string.
    /// @throws std::out_of_range if any token ID is not in vocabulary.
    std::string decode(const std::vector<int32_t>& tokens) const;

    // ── Special tokens & Metadata ─────────────────────────────────────────────

    /// @brief Total vocabulary size (e.g. 50257).
    size_t vocab_size() const noexcept { return token_to_id_.size(); }

    /// @brief Number of BPE merge rules loaded (e.g. 49992).
    size_t num_merges() const noexcept { return bpe_ranks_.size(); }

    /// @brief End-Of-Sequence token ID (<|endoftext|>, typically 50256).
    int32_t eos_token_id() const noexcept { return eos_token_id_; }

    /// @brief Check if a token ID exists in the vocabulary.
    bool has_token_id(int32_t id) const noexcept {
        return id >= 0 && static_cast<size_t>(id) < id_to_token_.size() && !id_to_token_[id].empty();
    }

    /// @brief Convert a single token ID to its string representation.
    const std::string& token_to_piece(int32_t id) const;

    /// @brief Convert a piece string to its token ID.
    int32_t piece_to_token(const std::string& piece) const;

    // ── Byte-level mapping utilities ──────────────────────────────────────────

    /// @brief Get the byte-to-unicode string mapping for a given raw byte (0..255).
    const std::string& byte_to_unicode(uint8_t b) const noexcept {
        return byte_to_unicode_[b];
    }

    /// @brief Look up raw byte for a unicode character string. Returns -1 if not found.
    int16_t unicode_to_byte(const std::string& uchar) const;

private:
    // ── Internal helpers ─────────────────────────────────────────────────────

    /// @brief Initialize the 256-byte bidirectional mapping table.
    void init_byte_mappings();

    /// @brief Load and parse vocab.json into encoder and decoder maps.
    void load_vocab(const std::string& vocab_file);

    /// @brief Load and parse merges.txt into bpe_ranks_ map.
    void load_merges(const std::string& merges_file);

    /// @brief Split raw UTF-8 text into pre-tokenization chunks matching the GPT-2 regex.
    std::vector<std::string> split_regex(const std::string& text) const;

    /// @brief Apply greedy BPE merges to a single word/chunk.
    std::vector<std::string> bpe(const std::string& word) const;

    // ── Data members ─────────────────────────────────────────────────────────

    std::unordered_map<std::string, int32_t> token_to_id_; ///< "ĠThe" -> 383
    std::vector<std::string> id_to_token_;                 ///< 383 -> "ĠThe"

    std::unordered_map<std::pair<std::string, std::string>, size_t, PairHash> bpe_ranks_;

    std::string byte_to_unicode_[256];                     ///< byte -> utf8 string
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;

    int32_t eos_token_id_ = 50256;                         ///< Default GPT-2 EOS token ID

    // Mutable cache for single-chunk BPE results to accelerate repeated words
    mutable std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;
};

} // namespace bmgpt
