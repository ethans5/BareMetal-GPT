#pragma once

/// @file WeightLoader.hpp
/// @brief Parser for the .safetensors binary format.
///
/// .safetensors layout:
///   [8 bytes] uint64_le  — byte length of the JSON header (N)
///   [N bytes] UTF-8 JSON — map: name -> { "dtype": "F32",
///                                          "shape": [...],
///                                          "data_offsets": [begin, end] }
///   [blob]    raw float32 data — buffers concatenated in header order
///
/// data_offsets are relative to the start of the blob (i.e., offset 8 + N).
///
/// Usage:
/// @code
///   WeightLoader loader("weights/micro_gpt2_weights.safetensors");
///   Tensor wte = loader.load("wte.weight");  // shape [256, 64]
/// @endcode

#include "Tensor.hpp"

#include <string>
#include <unordered_map>
#include <vector>
#include <cstddef>
#include <cstdint>

namespace bmgpt {

/// @class WeightLoader
/// @brief Lazy loader for .safetensors weight files.
///
/// The JSON header is parsed once at construction. Each call to load()
/// opens the file, seeks to the tensor's byte range, reads the float32
/// buffer, and returns a populated Tensor. File descriptors are not
/// kept open between calls (safe for large files, no fd leak).
class WeightLoader {
public:
    // -- Constructor ---------------------------------------------------------

    /// @brief Parse the .safetensors header and build the internal index.
    /// @param path  Absolute or relative path to a .safetensors file.
    /// @throws std::runtime_error if the file cannot be opened, the header
    ///         is malformed, or a tensor has an unsupported dtype (not F32).
    explicit WeightLoader(const std::string& path);

    // -- Tensor loading ------------------------------------------------------

    /// @brief Load a tensor by name from the weight file.
    /// @param name  Tensor key as it appears in the file header
    ///              (e.g. "wte.weight", "h.0.ln_1.weight").
    /// @return A new Tensor with the correct shape, filled with float32 data.
    /// @throws std::runtime_error if @p name is not found in the index.
    Tensor load(const std::string& name) const;

    // -- Index queries --------------------------------------------------------

    /// @brief Return true if the given key exists in the weight file.
    bool has(const std::string& name) const;

    /// @brief Return all tensor names contained in the file (sorted).
    std::vector<std::string> keys() const;

    /// @brief Number of tensors in the file (excludes __metadata__).
    std::size_t num_tensors() const noexcept { return meta_.size(); }

private:
    /// @brief Parsed metadata for one tensor entry.
    struct TensorMeta {
        std::vector<std::size_t> shape;  ///< Tensor dimensions.
        std::uint64_t            begin;  ///< Blob-relative byte start.
        std::uint64_t            end;    ///< Blob-relative byte end (exclusive).
    };

    std::string                                 filepath_;     ///< Path to the .safetensors file.
    std::uint64_t                               data_offset_;  ///< File offset (bytes) where the blob starts.
    std::unordered_map<std::string, TensorMeta> meta_;         ///< Parsed tensor index.
};

} // namespace bmgpt
