/// @file Tokenizer.cpp
/// @brief Implementation of GPT-2 Byte-Level BPE Tokenizer (Phase 5 — Tâche 1).

#include "baremetalgpt/Tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace bmgpt {

namespace {

// ── UTF-8 Encoding & Decoding Helpers ────────────────────────────────────────

std::string codepoint_to_utf8(uint32_t cp)
{
    std::string out;
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

/// @brief Decode next UTF-8 codepoint from text starting at index i. Advances i.
uint32_t decode_next_utf8(const std::string& text, size_t& i)
{
    if (i >= text.size()) return 0;
    const unsigned char c0 = static_cast<unsigned char>(text[i]);
    if (c0 <= 0x7F) {
        ++i;
        return c0;
    }
    if ((c0 & 0xE0) == 0xC0 && i + 1 < text.size()) {
        const unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
        i += 2;
        return ((c0 & 0x1F) << 6) | (c1 & 0x3F);
    }
    if ((c0 & 0xF0) == 0xE0 && i + 2 < text.size()) {
        const unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
        const unsigned char c2 = static_cast<unsigned char>(text[i + 2]);
        i += 3;
        return ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
    }
    if ((c0 & 0xF8) == 0xF0 && i + 3 < text.size()) {
        const unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
        const unsigned char c2 = static_cast<unsigned char>(text[i + 2]);
        const unsigned char c3 = static_cast<unsigned char>(text[i + 3]);
        i += 4;
        return ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
    }
    // Fallback on invalid sequence
    ++i;
    return c0;
}

/// @brief Categorize a codepoint into 'L' (Letter), 'N' (Number), 'S' (Space), 'O' (Other).
char get_unicode_category(uint32_t cp)
{
    if (cp < 128) {
        if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\f' || cp == '\v') {
            return 'S';
        }
        if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) {
            return 'L';
        }
        if (cp >= '0' && cp <= '9') {
            return 'N';
        }
        return 'O';
    }

    // Unicode whitespace
    if (cp == 0x00A0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
        cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000) {
        return 'S';
    }

    // Unicode numbers (common ranges)
    if ((cp >= 0x0660 && cp <= 0x0669) || (cp >= 0x06F0 && cp <= 0x06F9) ||
        (cp >= 0x0966 && cp <= 0x096F) || (cp >= 0x09E6 && cp <= 0x09EF) ||
        (cp >= 0x0A66 && cp <= 0x0A6F) || (cp >= 0x0AE6 && cp <= 0x0AEF) ||
        (cp >= 0x0B66 && cp <= 0x0B6F) || (cp >= 0x0BE6 && cp <= 0x0BEF) ||
        (cp >= 0x0C66 && cp <= 0x0C6F) || (cp >= 0x0CE6 && cp <= 0x0CEF) ||
        (cp >= 0x0D66 && cp <= 0x0D6F) || (cp >= 0x0E50 && cp <= 0x0E59) ||
        (cp >= 0x0ED0 && cp <= 0x0ED9) || (cp >= 0x0F20 && cp <= 0x0F29) ||
        (cp >= 0x1040 && cp <= 0x1049) || (cp >= 0x1090 && cp <= 0x1099) ||
        (cp >= 0xFF10 && cp <= 0xFF19) || (cp >= 0x2070 && cp <= 0x2079) ||
        (cp >= 0x2080 && cp <= 0x2089) || (cp >= 0x2460 && cp <= 0x249B)) {
        return 'N';
    }

    // Common Unicode Punctuation & Symbols (including Emojis)
    if ((cp >= 0x00A1 && cp <= 0x00BF && cp != 0x00AA && cp != 0x00BA) ||
        cp == 0x00D7 || cp == 0x00F7 ||
        (cp >= 0x2000 && cp <= 0x2BFF) ||
        (cp >= 0x2E00 && cp <= 0x2E7F) ||
        (cp >= 0x3001 && cp <= 0x303F) ||
        (cp >= 0xFE30 && cp <= 0xFE6F) ||
        (cp >= 0xFF01 && cp <= 0xFF0F) || (cp >= 0xFF1A && cp <= 0xFF20) ||
        (cp >= 0xFF3B && cp <= 0xFF40) || (cp >= 0xFF5B && cp <= 0xFF65) ||
        (cp >= 0xFFE0 && cp <= 0xFFEE) ||
        (cp >= 0x1F000 && cp <= 0x1FAFF)) {
        return 'O';
    }

    // Default for remaining non-ASCII scripts (Latin Extended, Greek, Cyrillic, CJK, etc.)
    return 'L';
}

} // namespace

// ── Constructor ──────────────────────────────────────────────────────────────

Tokenizer::Tokenizer(const std::string& vocab_file, const std::string& merges_file)
{
    init_byte_mappings();
    load_vocab(vocab_file);
    load_merges(merges_file);
}

// ── Byte-to-Unicode initialization ───────────────────────────────────────────

void Tokenizer::init_byte_mappings()
{
    // bs: '!' to '~' (33..126), '¡' to '¬' (161..172), '®' to 'ÿ' (174..255)
    std::vector<int> bs;
    bs.reserve(256);
    for (int b = 33; b <= 126; ++b) bs.push_back(b);
    for (int b = 161; b <= 172; ++b) bs.push_back(b);
    for (int b = 174; b <= 255; ++b) bs.push_back(b);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }

    for (size_t i = 0; i < bs.size(); ++i) {
        uint8_t byte_val = static_cast<uint8_t>(bs[i]);
        std::string u_str = codepoint_to_utf8(static_cast<uint32_t>(cs[i]));
        byte_to_unicode_[byte_val] = u_str;
        unicode_to_byte_[u_str] = byte_val;
    }
}

int16_t Tokenizer::unicode_to_byte(const std::string& uchar) const
{
    auto it = unicode_to_byte_.find(uchar);
    if (it != unicode_to_byte_.end()) {
        return static_cast<int16_t>(it->second);
    }
    return -1;
}

// ── Load vocab.json ──────────────────────────────────────────────────────────

void Tokenizer::load_vocab(const std::string& vocab_file)
{
    std::ifstream file(vocab_file, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Tokenizer: cannot open vocab file: " + vocab_file);
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    // Ultra-fast zero-dependency JSON parser for flat {"token": id, ...}
    const size_t len = content.size();
    size_t i = 0;

    // Find '{'
    while (i < len && content[i] != '{') ++i;
    if (i >= len) {
        throw std::runtime_error("Tokenizer: malformed vocab.json (missing '{')");
    }
    ++i;

    while (i < len) {
        // Find opening quote of key
        while (i < len && content[i] != '"' && content[i] != '}') ++i;
        if (i >= len || content[i] == '}') break;
        ++i; // skip opening quote

        // Parse key with escapes
        std::string key;
        while (i < len && content[i] != '"') {
            if (content[i] == '\\' && i + 1 < len) {
                char esc = content[i + 1];
                if (esc == '"' || esc == '\\' || esc == '/') {
                    key.push_back(esc);
                    i += 2;
                } else if (esc == 'n') {
                    key.push_back('\n');
                    i += 2;
                } else if (esc == 'r') {
                    key.push_back('\r');
                    i += 2;
                } else if (esc == 't') {
                    key.push_back('\t');
                    i += 2;
                } else if (esc == 'u' && i + 5 < len) {
                    // 4 hex digits: \uXXXX
                    std::string hex_str = content.substr(i + 2, 4);
                    uint32_t cp = static_cast<uint32_t>(std::stoul(hex_str, nullptr, 16));
                    key += codepoint_to_utf8(cp);
                    i += 6;
                } else {
                    key.push_back(content[i]);
                    ++i;
                }
            } else {
                key.push_back(content[i]);
                ++i;
            }
        }
        if (i < len && content[i] == '"') ++i; // skip closing quote

        // Find ':'
        while (i < len && content[i] != ':') ++i;
        if (i >= len) break;
        ++i; // skip ':'

        // Skip whitespace before number
        while (i < len && (std::isspace(static_cast<unsigned char>(content[i])))) ++i;

        // Parse integer ID
        size_t num_start = i;
        if (i < len && content[i] == '-') ++i;
        while (i < len && std::isdigit(static_cast<unsigned char>(content[i]))) ++i;
        int32_t id = std::stoi(content.substr(num_start, i - num_start));

        // Store
        token_to_id_[key] = id;
        if (static_cast<size_t>(id) >= id_to_token_.size()) {
            id_to_token_.resize(static_cast<size_t>(id) + 1);
        }
        id_to_token_[id] = key;

        // Skip until ',' or '}'
        while (i < len && content[i] != ',' && content[i] != '}') ++i;
        if (i < len && content[i] == ',') ++i;
    }

    if (token_to_id_.empty()) {
        throw std::runtime_error("Tokenizer: empty or invalid vocab.json");
    }

    // Set EOS token ID (<|endoftext|>)
    auto it = token_to_id_.find("<|endoftext|>");
    if (it != token_to_id_.end()) {
        eos_token_id_ = it->second;
    } else {
        eos_token_id_ = 50256;
    }
}

// ── Load merges.txt ──────────────────────────────────────────────────────────

void Tokenizer::load_merges(const std::string& merges_file)
{
    std::ifstream file(merges_file, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Tokenizer: cannot open merges file: " + merges_file);
    }

    std::string line;
    size_t rank = 0;
    while (std::getline(file, line)) {
        // Strip carriage return if present
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Find space separating the pair
        size_t space_pos = line.find(' ');
        if (space_pos != std::string::npos && space_pos + 1 < line.size()) {
            std::string first = line.substr(0, space_pos);
            std::string second = line.substr(space_pos + 1);
            bpe_ranks_[{std::move(first), std::move(second)}] = rank++;
        }
    }

    if (bpe_ranks_.empty()) {
        throw std::runtime_error("Tokenizer: empty or invalid merges.txt");
    }
}

// ── Regex pre-tokenization ───────────────────────────────────────────────────

std::vector<std::string> Tokenizer::split_regex(const std::string& text) const
{
    std::vector<std::string> pieces;
    if (text.empty()) return pieces;

    // Decode codepoints with byte offsets
    struct CPInfo {
        uint32_t cp;
        size_t byte_offset;
        char cat;
    };

    std::vector<CPInfo> cps;
    cps.reserve(text.size());
    size_t byte_idx = 0;
    while (byte_idx < text.size()) {
        size_t start_off = byte_idx;
        uint32_t cp = decode_next_utf8(text, byte_idx);
        cps.push_back({cp, start_off, get_unicode_category(cp)});
    }

    const size_t N = cps.size();
    size_t i = 0;

    static const std::vector<std::string> contractions = {
        "'s", "'t", "'re", "'ve", "'m", "'ll", "'d"
    };

    while (i < N) {
        size_t start_byte = cps[i].byte_offset;

        // Branch 1: Contractions
        bool matched_contraction = false;
        for (const auto& cont : contractions) {
            if (text.compare(start_byte, cont.size(), cont) == 0) {
                pieces.push_back(cont);
                // Advance i by number of codepoints in cont (all ASCII, so length == byte size)
                i += cont.size();
                matched_contraction = true;
                break;
            }
        }
        if (matched_contraction) continue;

        // Check if current token starts with an optional space ' '
        bool has_space = (cps[i].cp == ' ');
        size_t next_idx = i + (has_space ? 1 : 0);

        if (next_idx < N) {
            char next_cat = cps[next_idx].cat;

            // Branch 2: ?\p{L}+
            if (next_cat == 'L') {
                size_t end = next_idx + 1;
                while (end < N && cps[end].cat == 'L') ++end;
                size_t end_byte = (end < N) ? cps[end].byte_offset : text.size();
                pieces.push_back(text.substr(start_byte, end_byte - start_byte));
                i = end;
                continue;
            }

            // Branch 3: ?\p{N}+
            if (next_cat == 'N') {
                size_t end = next_idx + 1;
                while (end < N && cps[end].cat == 'N') ++end;
                size_t end_byte = (end < N) ? cps[end].byte_offset : text.size();
                pieces.push_back(text.substr(start_byte, end_byte - start_byte));
                i = end;
                continue;
            }

            // Branch 4: ?[^\s\p{L}\p{N}]+
            if (next_cat == 'O') {
                size_t end = next_idx + 1;
                while (end < N && cps[end].cat == 'O') ++end;
                size_t end_byte = (end < N) ? cps[end].byte_offset : text.size();
                pieces.push_back(text.substr(start_byte, end_byte - start_byte));
                i = end;
                continue;
            }
        }

        // Branch 5 & 6: \s+(?!\S)|\s+
        size_t end = i;
        while (end < N && cps[end].cat == 'S') ++end;

        if (end > i) {
            // If whitespace run is immediately followed by a non-whitespace character,
            // and the last whitespace in the run is ' ', leave that last ' ' for the next token
            if (end < N && cps[end].cat != 'S' && cps[end - 1].cp == ' ') {
                if (end - 1 > i) {
                    size_t end_byte = cps[end - 1].byte_offset;
                    pieces.push_back(text.substr(start_byte, end_byte - start_byte));
                    i = end - 1;
                    continue;
                }
            }
            size_t end_byte = (end < N) ? cps[end].byte_offset : text.size();
            pieces.push_back(text.substr(start_byte, end_byte - start_byte));
            i = end;
            continue;
        }

        // Fallback: single codepoint
        size_t end_byte = (i + 1 < N) ? cps[i + 1].byte_offset : text.size();
        pieces.push_back(text.substr(start_byte, end_byte - start_byte));
        ++i;
    }

    return pieces;
}

// ── BPE Algorithm ────────────────────────────────────────────────────────────

std::vector<std::string> Tokenizer::bpe(const std::string& word) const
{
    auto it_cache = bpe_cache_.find(word);
    if (it_cache != bpe_cache_.end()) {
        return it_cache->second;
    }

    // Split word into initial Unicode character symbols
    std::vector<std::string> symbols;
    symbols.reserve(word.size());
    size_t i = 0;
    while (i < word.size()) {
        size_t start = i;
        decode_next_utf8(word, i);
        symbols.push_back(word.substr(start, i - start));
    }

    if (symbols.size() <= 1) {
        bpe_cache_[word] = symbols;
        return symbols;
    }

    while (symbols.size() > 1) {
        // Find adjacent pair with minimum rank in bpe_ranks_
        size_t min_rank = std::numeric_limits<size_t>::max();
        size_t best_idx = symbols.size();

        for (size_t j = 0; j < symbols.size() - 1; ++j) {
            auto it = bpe_ranks_.find({symbols[j], symbols[j + 1]});
            if (it != bpe_ranks_.end() && it->second < min_rank) {
                min_rank = it->second;
                best_idx = j;
            }
        }

        if (best_idx == symbols.size()) {
            break; // No merge possible
        }

        const std::string p1 = symbols[best_idx];
        const std::string p2 = symbols[best_idx + 1];

        // Merge all occurrences of (p1, p2) in symbols
        std::vector<std::string> new_symbols;
        new_symbols.reserve(symbols.size());

        for (size_t j = 0; j < symbols.size(); ++j) {
            if (j + 1 < symbols.size() && symbols[j] == p1 && symbols[j + 1] == p2) {
                new_symbols.push_back(p1 + p2);
                ++j; // Skip second element
            } else {
                new_symbols.push_back(symbols[j]);
            }
        }
        symbols = std::move(new_symbols);
    }

    bpe_cache_[word] = symbols;
    return symbols;
}

// ── Tokenization & Detokenization ─────────────────────────────────────────────

std::vector<int32_t> Tokenizer::encode(const std::string& text) const
{
    if (text.empty()) return {};

    // Special token exact match optimization
    if (text == "<|endoftext|>") {
        return {eos_token_id_};
    }

    std::vector<std::string> pieces = split_regex(text);
    std::vector<int32_t> tokens;
    tokens.reserve(pieces.size() * 2);

    for (const auto& piece : pieces) {
        // Check if piece itself is a special token
        if (piece == "<|endoftext|>") {
            tokens.push_back(eos_token_id_);
            continue;
        }

        // 1. Translate raw UTF-8 bytes to unicode characters
        std::string translated;
        translated.reserve(piece.size() * 2);
        for (unsigned char b : piece) {
            translated += byte_to_unicode_[b];
        }

        // 2. Apply BPE merges
        std::vector<std::string> bpe_pieces = bpe(translated);

        // 3. Lookup token IDs
        for (const auto& sym : bpe_pieces) {
            auto it = token_to_id_.find(sym);
            if (it != token_to_id_.end()) {
                tokens.push_back(it->second);
            } else {
                throw std::runtime_error("Tokenizer::encode: unknown symbol: " + sym);
            }
        }
    }

    return tokens;
}

std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const
{
    if (tokens.empty()) return "";

    std::string byte_buffer;
    for (int32_t id : tokens) {
        if (!has_token_id(id)) {
            throw std::out_of_range("Tokenizer::decode: token ID " + std::to_string(id) + " out of range");
        }
        const std::string& piece = id_to_token_[id];

        // Decode each unicode character back to its raw byte
        size_t i = 0;
        while (i < piece.size()) {
            size_t start = i;
            decode_next_utf8(piece, i);
            std::string uchar = piece.substr(start, i - start);

            auto it = unicode_to_byte_.find(uchar);
            if (it != unicode_to_byte_.end()) {
                byte_buffer.push_back(static_cast<char>(it->second));
            } else {
                // Special token or unmapped character: append as-is
                byte_buffer += uchar;
            }
        }
    }

    return byte_buffer;
}

const std::string& Tokenizer::token_to_piece(int32_t id) const
{
    if (!has_token_id(id)) {
        throw std::out_of_range("Tokenizer::token_to_piece: invalid token ID " + std::to_string(id));
    }
    return id_to_token_[id];
}

int32_t Tokenizer::piece_to_token(const std::string& piece) const
{
    auto it = token_to_id_.find(piece);
    if (it == token_to_id_.end()) {
        throw std::out_of_range("Tokenizer::piece_to_token: piece not in vocab: " + piece);
    }
    return it->second;
}

} // namespace bmgpt
