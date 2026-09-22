/// @file WeightLoader.cpp
/// @brief Implementation of the .safetensors parser.
///
/// The .safetensors format is:
///   [8 bytes]  uint64_le : byte size of the JSON header (N)
///   [N bytes]  UTF-8 JSON: { "<name>": {"dtype":"F32","shape":[...],"data_offsets":[b,e]}, ... }
///   [blob]     raw little-endian float32 data
///
/// This parser handles the subset of JSON used by safetensors:
///   - String keys/values (no escape sequences beyond those in tensor names)
///   - Integer arrays for shape and data_offsets
///   - F32 dtype only
///
/// Design: no external JSON library — the header grammar is simple enough
/// to parse with standard C++ string operations.

#include "baremetalgpt/WeightLoader.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bmgpt {

// ---------------------------------------------------------------------------
// Minimal JSON utilities (private to this TU)
// ---------------------------------------------------------------------------

namespace {

/// @brief Skip whitespace characters in @p s starting at @p pos.
static void skip_ws(const std::string& s, std::size_t& pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' ||
                               s[pos] == '\n' || s[pos] == '\r')) {
        ++pos;
    }
}

/// @brief Consume a specific character, throw if not found.
static void expect(const std::string& s, std::size_t& pos, char c) {
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != c) {
        throw std::runtime_error(
            std::string("[WeightLoader] JSON parse error: expected '") + c +
            "' at position " + std::to_string(pos));
    }
    ++pos;
}

/// @brief Parse a JSON string (assumes cursor is at opening '"').
static std::string parse_string(const std::string& s, std::size_t& pos) {
    skip_ws(s, pos);
    expect(s, pos, '"');
    std::string result;
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\') {
            ++pos; // skip backslash
            if (pos < s.size()) {
                result += s[pos++];
            }
        } else {
            result += s[pos++];
        }
    }
    expect(s, pos, '"');
    return result;
}

/// @brief Parse a JSON unsigned integer.
static std::uint64_t parse_uint(const std::string& s, std::size_t& pos) {
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] < '0' || s[pos] > '9') {
        throw std::runtime_error(
            "[WeightLoader] JSON parse error: expected integer at position " +
            std::to_string(pos));
    }
    std::uint64_t val = 0;
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') {
        val = val * 10 + static_cast<std::uint64_t>(s[pos++] - '0');
    }
    return val;
}

/// @brief Parse a JSON array of unsigned integers: [n0, n1, ...].
static std::vector<std::uint64_t> parse_uint_array(const std::string& s, std::size_t& pos) {
    expect(s, pos, '[');
    std::vector<std::uint64_t> arr;
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == ']') {
        ++pos;
        return arr;
    }
    arr.push_back(parse_uint(s, pos));
    skip_ws(s, pos);
    while (pos < s.size() && s[pos] == ',') {
        ++pos;
        arr.push_back(parse_uint(s, pos));
        skip_ws(s, pos);
    }
    expect(s, pos, ']');
    return arr;
}

/// @brief Skip any JSON value (string, number, object, array, bool, null).
/// Used to skip unknown keys inside tensor entries (e.g. "__metadata__").
static void skip_value(const std::string& s, std::size_t& pos);

static void skip_object(const std::string& s, std::size_t& pos) {
    expect(s, pos, '{');
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == '}') { ++pos; return; }
    while (pos < s.size()) {
        parse_string(s, pos);    // key
        skip_ws(s, pos);
        expect(s, pos, ':');
        skip_value(s, pos);      // value
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ',') { ++pos; skip_ws(s, pos); }
        else break;
    }
    expect(s, pos, '}');
}

static void skip_array(const std::string& s, std::size_t& pos) {
    expect(s, pos, '[');
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == ']') { ++pos; return; }
    while (pos < s.size()) {
        skip_value(s, pos);
        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ',') { ++pos; skip_ws(s, pos); }
        else break;
    }
    expect(s, pos, ']');
}

static void skip_value(const std::string& s, std::size_t& pos) {
    skip_ws(s, pos);
    if (pos >= s.size()) return;
    char c = s[pos];
    if (c == '"') { parse_string(s, pos); }
    else if (c == '{') { skip_object(s, pos); }
    else if (c == '[') { skip_array(s, pos); }
    else if (c == 't') { pos += 4; }  // true
    else if (c == 'f') { pos += 5; }  // false
    else if (c == 'n') { pos += 4; }  // null
    else {
        // number (integer or float)
        if (c == '-') ++pos;
        while (pos < s.size() && (s[pos] >= '0' && s[pos] <= '9')) ++pos;
        if (pos < s.size() && s[pos] == '.') {
            ++pos;
            while (pos < s.size() && (s[pos] >= '0' && s[pos] <= '9')) ++pos;
        }
        if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
            ++pos;
            if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) ++pos;
            while (pos < s.size() && (s[pos] >= '0' && s[pos] <= '9')) ++pos;
        }
    }
}

/// @brief Parse a single tensor entry object:
///        { "dtype": "F32", "shape": [...], "data_offsets": [b, e] }
/// @return Parsed TensorMeta.
struct RawMeta {
    std::vector<std::uint64_t> shape;
    std::uint64_t begin = 0;
    std::uint64_t end   = 0;
};

static RawMeta parse_tensor_entry(const std::string& s, std::size_t& pos) {
    RawMeta m;
    bool    has_dtype   = false;
    bool    has_shape   = false;
    bool    has_offsets = false;

    expect(s, pos, '{');
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == '}') { ++pos; return m; }

    while (pos < s.size()) {
        std::string key = parse_string(s, pos);
        skip_ws(s, pos);
        expect(s, pos, ':');
        skip_ws(s, pos);

        if (key == "dtype") {
            std::string dtype = parse_string(s, pos);
            if (dtype != "F32") {
                throw std::runtime_error(
                    "[WeightLoader] Unsupported dtype: \"" + dtype +
                    "\" (only F32 is supported)");
            }
            has_dtype = true;
        } else if (key == "shape") {
            m.shape     = parse_uint_array(s, pos);
            has_shape   = true;
        } else if (key == "data_offsets") {
            auto off    = parse_uint_array(s, pos);
            if (off.size() != 2) {
                throw std::runtime_error(
                    "[WeightLoader] data_offsets must have exactly 2 elements");
            }
            m.begin     = off[0];
            m.end       = off[1];
            has_offsets = true;
        } else {
            skip_value(s, pos);
        }

        skip_ws(s, pos);
        if (pos < s.size() && s[pos] == ',') { ++pos; skip_ws(s, pos); }
        else break;
    }
    expect(s, pos, '}');

    if (!has_dtype || !has_shape || !has_offsets) {
        throw std::runtime_error(
            "[WeightLoader] Incomplete tensor entry in JSON header "
            "(missing dtype, shape, or data_offsets)");
    }
    return m;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// WeightLoader constructor — parse header
// ---------------------------------------------------------------------------

WeightLoader::WeightLoader(const std::string& path)
    : filepath_(path), data_offset_(0)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error(
            "[WeightLoader] Cannot open file: \"" + path + "\"");
    }

    // 1. Read 8-byte header length (little-endian uint64)
    std::uint64_t header_len = 0;
    file.read(reinterpret_cast<char*>(&header_len), 8);
    if (!file || file.gcount() != 8) {
        throw std::runtime_error(
            "[WeightLoader] Failed to read header length from: \"" + path + "\"");
    }

    // 2. Read JSON header string
    std::string json_str(static_cast<std::size_t>(header_len), '\0');
    file.read(json_str.data(), static_cast<std::streamsize>(header_len));
    if (!file || static_cast<std::uint64_t>(file.gcount()) != header_len) {
        throw std::runtime_error(
            "[WeightLoader] Truncated JSON header in: \"" + path + "\"");
    }

    data_offset_ = 8u + header_len;  // blob starts right after header

    // 3. Parse the top-level JSON object
    std::size_t pos = 0;
    expect(json_str, pos, '{');
    skip_ws(json_str, pos);
    if (pos < json_str.size() && json_str[pos] == '}') {
        return; // empty file
    }

    while (pos < json_str.size()) {
        std::string name = parse_string(json_str, pos);
        skip_ws(json_str, pos);
        expect(json_str, pos, ':');
        skip_ws(json_str, pos);

        if (name == "__metadata__") {
            // Skip the metadata object (not a tensor entry)
            skip_object(json_str, pos);
        } else {
            RawMeta rm = parse_tensor_entry(json_str, pos);

            TensorMeta tm;
            tm.shape.reserve(rm.shape.size());
            for (auto d : rm.shape) {
                tm.shape.push_back(static_cast<std::size_t>(d));
            }
            tm.begin = rm.begin;
            tm.end   = rm.end;

            // Validate: byte range must equal numel * sizeof(float)
            std::size_t numel = 1;
            for (auto d : tm.shape) { numel *= d; }
            std::uint64_t expected_bytes =
                static_cast<std::uint64_t>(numel) * sizeof(float);
            if (tm.end - tm.begin != expected_bytes) {
                throw std::runtime_error(
                    "[WeightLoader] Byte range mismatch for tensor \"" + name +
                    "\": data_offsets span " +
                    std::to_string(tm.end - tm.begin) +
                    " bytes, but shape implies " +
                    std::to_string(expected_bytes) + " bytes (numel=" +
                    std::to_string(numel) + ")");
            }

            meta_.emplace(name, std::move(tm));
        }

        skip_ws(json_str, pos);
        if (pos < json_str.size() && json_str[pos] == ',') {
            ++pos;
            skip_ws(json_str, pos);
        } else {
            break;
        }
    }
    // Consume closing '}'
    skip_ws(json_str, pos);
    if (pos < json_str.size() && json_str[pos] == '}') {
        ++pos;
    }
}

// ---------------------------------------------------------------------------
// WeightLoader::load
// ---------------------------------------------------------------------------

Tensor WeightLoader::load(const std::string& name) const {
    auto it = meta_.find(name);
    if (it == meta_.end()) {
        throw std::runtime_error(
            "[WeightLoader] Tensor not found: \"" + name + "\"");
    }
    const TensorMeta& tm = it->second;

    // Compute element count
    std::size_t numel = 1;
    for (auto d : tm.shape) { numel *= d; }

    // Open file and seek to tensor blob
    std::ifstream file(filepath_, std::ios::binary);
    if (!file) {
        throw std::runtime_error(
            "[WeightLoader] Cannot re-open file: \"" + filepath_ + "\"");
    }
    std::uint64_t abs_offset = data_offset_ + tm.begin;
    file.seekg(static_cast<std::streamoff>(abs_offset));
    if (!file) {
        throw std::runtime_error(
            "[WeightLoader] Seek failed for tensor \"" + name + "\"");
    }

    // Read float32 buffer
    std::vector<float> buf(numel);
    file.read(reinterpret_cast<char*>(buf.data()),
              static_cast<std::streamsize>(numel * sizeof(float)));
    if (!file || static_cast<std::size_t>(file.gcount()) != numel * sizeof(float)) {
        throw std::runtime_error(
            "[WeightLoader] Short read for tensor \"" + name + "\"");
    }

    return Tensor(tm.shape, std::move(buf));
}

// ---------------------------------------------------------------------------
// WeightLoader::has / keys
// ---------------------------------------------------------------------------

bool WeightLoader::has(const std::string& name) const {
    return meta_.count(name) > 0;
}

std::vector<std::string> WeightLoader::keys() const {
    std::vector<std::string> ks;
    ks.reserve(meta_.size());
    for (const auto& kv : meta_) {
        ks.push_back(kv.first);
    }
    std::sort(ks.begin(), ks.end());
    return ks;
}

} // namespace bmgpt
