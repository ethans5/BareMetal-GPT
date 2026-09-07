#pragma once

/// @file Tensor.hpp
/// @brief Core N-dimensional tensor class for the BareMetal-GPT inference engine.
///
/// This tensor is the fundamental data structure used throughout the project.
/// It stores dense float data in row-major (C-contiguous) order and provides
/// bounds-checked element access, element-wise arithmetic, and shape utilities.
///
/// Design choices:
///   - float32 only (matches GPT-2 weight precision).
///   - Row-major layout to mirror NumPy/PyTorch default ordering, making
///     weight loading from .safetensors straightforward.
///   - No GPU support — this is a CPU-only, from-scratch implementation.

#include <vector>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace bmgpt {

/// @class Tensor
/// @brief N-dimensional tensor with row-major (C-contiguous) storage.
///
/// Holds a flat float buffer alongside shape and precomputed stride metadata.
/// Strides are expressed in number of elements (not bytes).
///
/// Example usage:
/// @code
///   Tensor mat({2, 3});            // 2x3 matrix, zero-initialized
///   mat.at({0, 1}) = 3.14f;        // set element at row 0, col 1
///   float v = mat.at({0, 1});      // read it back
/// @endcode
class Tensor {
private:
    std::vector<size_t> shape_;    ///< Dimensions of the tensor (e.g. {batch, rows, cols}).
    std::vector<size_t> strides_;  ///< Number of elements to skip per dimension (row-major).
    std::vector<float>  data_;     ///< Flat contiguous buffer holding all elements.

    /// @brief Compute row-major strides from a shape vector.
    /// @param shape  Dimension sizes (e.g. {2, 3, 4}).
    /// @return Stride vector such that strides[i] = product of shape[i+1 .. end].
    ///
    /// For shape {2, 3, 4} the strides are {12, 4, 1}.
    static std::vector<size_t> compute_strides(const std::vector<size_t>& shape);

    /// @brief Convert multi-dimensional indices into a flat offset.
    /// @param indices  One index per dimension.
    /// @return Flat index into data_.
    /// @throws std::out_of_range if the number of indices doesn't match ndim()
    ///         or any index is out of bounds.
    size_t flatten_index(const std::vector<size_t>& indices) const;

public:
    // ── Constructors ────────────────────────────────────────────────

    /// @brief Default constructor — creates an empty, shapeless tensor.
    Tensor() = default;

    /// @brief Allocate a zero-initialized tensor with the given shape.
    /// @param shape  Desired dimensions (e.g. {batch, seq_len, d_model}).
    ///
    /// The total number of elements is the product of all dimensions.
    explicit Tensor(std::vector<size_t> shape);

    /// @brief Create a tensor with the given shape and pre-filled data.
    /// @param shape  Desired dimensions.
    /// @param data   Flat data buffer (row-major order). Its size must equal
    ///               the product of all dimensions in @p shape.
    /// @throws std::invalid_argument if data.size() != product(shape).
    Tensor(std::vector<size_t> shape, std::vector<float> data);

    // ── Shape & metadata queries ────────────────────────────────────

    /// @brief Return the shape vector (read-only).
    const std::vector<size_t>& shape() const noexcept { return shape_; }

    /// @brief Return the strides vector (read-only, in number of elements).
    const std::vector<size_t>& strides() const noexcept { return strides_; }

    /// @brief Number of dimensions (0 for a default-constructed tensor).
    size_t ndim() const noexcept { return shape_.size(); }

    /// @brief Size along a specific dimension.
    /// @param d  Dimension index (0-based).
    /// @throws std::out_of_range if d >= ndim().
    size_t size(size_t d) const;

    /// @brief Total number of elements in the tensor.
    size_t numel() const noexcept { return data_.size(); }

    // ── Raw data access ─────────────────────────────────────────────

    /// @brief Mutable pointer to the underlying flat buffer.
    /// Useful for bulk operations, BLAS calls, or weight loading.
    float* data() noexcept { return data_.data(); }

    /// @brief Const pointer to the underlying flat buffer.
    const float* data() const noexcept { return data_.data(); }

    // ── Element access (bounds-checked) ─────────────────────────────

    /// @brief Access an element by multi-dimensional index (initializer_list).
    /// @param indices  Brace-enclosed indices, e.g. {row, col}.
    /// @throws std::out_of_range on invalid indices.
    float&       at(std::initializer_list<size_t> indices);
    const float& at(std::initializer_list<size_t> indices) const;

    /// @brief Access an element by multi-dimensional index (vector).
    /// @param indices  Index vector — useful when dimensions are determined at runtime.
    /// @throws std::out_of_range on invalid indices.
    float&       at(const std::vector<size_t>& indices);
    const float& at(const std::vector<size_t>& indices) const;

    // ── Arithmetic operations ───────────────────────────────────────

    /// @brief Element-wise addition. Both tensors must have the same shape.
    /// @param other  Right-hand operand.
    /// @return A new tensor containing the element-wise sum.
    /// @throws std::invalid_argument if shapes don't match.
    Tensor operator+(const Tensor& other) const;

    // ── Shape transformations ───────────────────────────────────────

    /// @brief Return a new tensor with the same data but a different shape.
    /// @param new_shape  Desired dimensions. The product of all dimensions
    ///                   must equal numel() (total element count is invariant).
    /// @return A new tensor with a copy of the flat data under the new shape.
    ///         The data buffer is duplicated (not a view).
    /// @throws std::invalid_argument if product(new_shape) != numel().
    Tensor reshape(std::vector<size_t> new_shape) const;

    /// @brief Swap two dimensions, returning a new contiguous tensor.
    /// @param dim0  First dimension to swap (0-based).
    /// @param dim1  Second dimension to swap (0-based).
    /// @return A new tensor with dims dim0 and dim1 exchanged and data
    ///         physically rearranged (always contiguous).
    ///         If dim0 == dim1, returns a copy (identity transpose).
    /// @throws std::out_of_range if dim0 >= ndim() or dim1 >= ndim().
    Tensor transpose(size_t dim0, size_t dim1) const;

    // ── Debug utilities ─────────────────────────────────────────────

    /// @brief Human-readable shape string, e.g. "[2, 3, 4]".
    std::string shape_to_string() const;
};

} // namespace bmgpt
