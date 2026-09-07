#pragma once

/// @file matmul.hpp
/// @brief Matrix multiplication for 2D tensors (Phase 1).
///
/// Implemented as a free function rather than a Tensor method because
/// matmul is a derived operation that combines two tensors of potentially
/// different shapes — it goes beyond the "container + basic ops" scope of
/// the Tensor class itself.  Keeping it separate also makes the Phase 6
/// optimisation work (tiling, threading, SIMD) self-contained.

#include "baremetalgpt/Tensor.hpp"

namespace bmgpt {

/// @brief Multiply two 2D tensors (matrices).
/// @param a  Left operand of shape [M, K].
/// @param b  Right operand of shape [K, N].
/// @return   Result tensor of shape [M, N].
/// @throws std::invalid_argument if either tensor is not 2D, or if
///         inner dimensions don't match (a.size(1) != b.size(0)).
Tensor matmul(const Tensor& a, const Tensor& b);

} // namespace bmgpt
