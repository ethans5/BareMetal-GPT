#include "baremetalgpt/matmul.hpp"

#include <stdexcept>
#include <string>

namespace bmgpt {

Tensor matmul(const Tensor& a, const Tensor& b) {
    // ── Validation: 2D strict ──────────────────────────────────────
    if (a.ndim() != 2) {
        throw std::invalid_argument(
            "matmul — left operand must be 2D, got " +
            a.shape_to_string());
    }
    if (b.ndim() != 2) {
        throw std::invalid_argument(
            "matmul — right operand must be 2D, got " +
            b.shape_to_string());
    }
    if (a.size(1) != b.size(0)) {
        throw std::invalid_argument(
            "matmul — inner dimension mismatch: " +
            a.shape_to_string() + " vs " + b.shape_to_string());
    }

    const size_t M = a.size(0);
    const size_t K = a.size(1);
    const size_t N = b.size(1);

    Tensor result({M, N});  // zero-initialized

    const float* a_data = a.data();
    const float* b_data = b.data();
    float*       r_data = result.data();

    // i,k,j loop order for cache-friendliness on row-major layout:
    //   - a_data[i*K + k] is a scalar hoisted out of the inner loop
    //   - b_data[k*N + j] is sequential in j (stride 1 — cache-friendly)
    //   - r_data[i*N + j] is sequential in j (stride 1 — cache-friendly)
    // This avoids the column-strided access pattern of the naive i,j,k order.
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            const float a_ik = a_data[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                r_data[i * N + j] += a_ik * b_data[k * N + j];
            }
        }
    }

    return result;
}

} // namespace bmgpt
