#include "baremetalgpt/Tensor.hpp"

#include <numeric>    // std::accumulate
#include <sstream>    // std::ostringstream
#include <algorithm>  // std::copy
#include <stdexcept>

namespace bmgpt {

// ── Private helpers ─────────────────────────────────────────────────

std::vector<size_t> Tensor::compute_strides(const std::vector<size_t>& shape) {
    // Row-major (C-contiguous) strides.
    // For shape {2, 3, 4} → strides {12, 4, 1}.
    std::vector<size_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

size_t Tensor::flatten_index(const std::vector<size_t>& indices) const {
    if (indices.size() != ndim()) {
        throw std::out_of_range(
            "Tensor::at — expected " + std::to_string(ndim()) +
            " indices, got " + std::to_string(indices.size()));
    }
    size_t flat = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range(
                "Tensor::at — index " + std::to_string(indices[i]) +
                " out of range for dimension " + std::to_string(i) +
                " (size " + std::to_string(shape_[i]) + ")");
        }
        flat += indices[i] * strides_[i];
    }
    return flat;
}

// ── Constructors ────────────────────────────────────────────────────

Tensor::Tensor(std::vector<size_t> shape)
    : shape_(std::move(shape))
    , strides_(compute_strides(shape_))
{
    size_t n = 1;
    for (size_t s : shape_) n *= s;
    data_.resize(n, 0.0f);
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<float> data)
    : shape_(std::move(shape))
    , strides_(compute_strides(shape_))
    , data_(std::move(data))
{
    size_t expected = 1;
    for (size_t s : shape_) expected *= s;
    if (data_.size() != expected) {
        throw std::invalid_argument(
            "Tensor — data size (" + std::to_string(data_.size()) +
            ") does not match shape (expected " + std::to_string(expected) + ")");
    }
}

// ── Shape queries ───────────────────────────────────────────────────

size_t Tensor::size(size_t d) const {
    if (d >= ndim()) {
        throw std::out_of_range(
            "Tensor::size — dimension " + std::to_string(d) +
            " out of range (ndim = " + std::to_string(ndim()) + ")");
    }
    return shape_[d];
}

// ── Element access ──────────────────────────────────────────────────

float& Tensor::at(std::initializer_list<size_t> indices) {
    return data_[flatten_index(std::vector<size_t>(indices))];
}

const float& Tensor::at(std::initializer_list<size_t> indices) const {
    return data_[flatten_index(std::vector<size_t>(indices))];
}

float& Tensor::at(const std::vector<size_t>& indices) {
    return data_[flatten_index(indices)];
}

const float& Tensor::at(const std::vector<size_t>& indices) const {
    return data_[flatten_index(indices)];
}

// ── Arithmetic ──────────────────────────────────────────────────────

Tensor Tensor::operator+(const Tensor& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument(
            "Tensor::operator+ — shape mismatch: " +
            shape_to_string() + " vs " + other.shape_to_string());
    }
    Tensor result(shape_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

// ── Shape transformations ───────────────────────────────────────

Tensor Tensor::reshape(std::vector<size_t> new_shape) const {
    // Invariant: product(new_shape) must equal numel().
    size_t new_numel = 1;
    for (size_t s : new_shape) new_numel *= s;
    if (new_numel != numel()) {
        throw std::invalid_argument(
            "Tensor::reshape — cannot reshape " + shape_to_string() +
            " (" + std::to_string(numel()) + " elements) into shape with " +
            std::to_string(new_numel) + " elements");
    }
    // Copies the flat data buffer under a new shape and strides.
    // Not truly zero-copy (would require shared ownership via shared_ptr),
    // but acceptable for Phase 1 — reshape cost is negligible vs matmul.
    return Tensor(std::move(new_shape), std::vector<float>(data_));
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const {
    if (dim0 >= ndim()) {
        throw std::out_of_range(
            "Tensor::transpose — dim0 (" + std::to_string(dim0) +
            ") out of range (ndim = " + std::to_string(ndim()) + ")");
    }
    if (dim1 >= ndim()) {
        throw std::out_of_range(
            "Tensor::transpose — dim1 (" + std::to_string(dim1) +
            ") out of range (ndim = " + std::to_string(ndim()) + ")");
    }

    // Identity transpose — just copy.
    if (dim0 == dim1) {
        return Tensor(shape_, std::vector<float>(data_));
    }

    // Build destination shape: swap dim0 and dim1.
    std::vector<size_t> dst_shape = shape_;
    std::swap(dst_shape[dim0], dst_shape[dim1]);

    Tensor result(dst_shape);

    // Precompute destination strides for direct flat-offset calculation.
    const auto& dst_strides = result.strides();
    const size_t n = ndim();
    const size_t total = numel();

    // Odometer: iterate through all source indices in row-major order.
    std::vector<size_t> src_idx(n, 0);

    for (size_t flat_src = 0; flat_src < total; ++flat_src) {
        // Compute destination flat offset with swapped dimensions.
        size_t flat_dst = 0;
        for (size_t d = 0; d < n; ++d) {
            // Map source dimension d to the corresponding destination dimension.
            size_t dst_d = d;
            if (d == dim0) dst_d = dim1;
            else if (d == dim1) dst_d = dim0;
            flat_dst += src_idx[d] * dst_strides[dst_d];
        }

        result.data_[flat_dst] = data_[flat_src];

        // Advance odometer (increment from last dimension, carry on overflow).
        for (int d = static_cast<int>(n) - 1; d >= 0; --d) {
            if (++src_idx[d] < shape_[d]) break;
            src_idx[d] = 0;
        }
    }

    return result;
}

// ── Debug ───────────────────────────────────────────────────────────

std::string Tensor::shape_to_string() const {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << "]";
    return oss.str();
}

} // namespace bmgpt
