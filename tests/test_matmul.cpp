#include <gtest/gtest.h>
#include "baremetalgpt/matmul.hpp"

// ── Basic correctness ───────────────────────────────────────────────

TEST(MatMulTest, Square2x2) {
    // [1 2] * [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
    // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
    bmgpt::Tensor a({2, 2}, {1, 2, 3, 4});
    bmgpt::Tensor b({2, 2}, {5, 6, 7, 8});
    bmgpt::Tensor c = bmgpt::matmul(a, b);

    EXPECT_EQ(c.shape(), (std::vector<size_t>{2, 2}));
    EXPECT_FLOAT_EQ(c.at({0, 0}), 19.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 22.0f);
    EXPECT_FLOAT_EQ(c.at({1, 0}), 43.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 50.0f);
}

TEST(MatMulTest, NonSquare) {
    // [1 2 3] * [7  8 ] = [1*7+2*9+3*11   1*8+2*10+3*12] = [ 58  64]
    // [4 5 6]   [9  10]   [4*7+5*9+6*11   4*8+5*10+6*12]   [139 154]
    //           [11 12]
    bmgpt::Tensor a({2, 3}, {1, 2, 3, 4, 5, 6});
    bmgpt::Tensor b({3, 2}, {7, 8, 9, 10, 11, 12});
    bmgpt::Tensor c = bmgpt::matmul(a, b);

    EXPECT_EQ(c.shape(), (std::vector<size_t>{2, 2}));
    EXPECT_FLOAT_EQ(c.at({0, 0}), 58.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 64.0f);
    EXPECT_FLOAT_EQ(c.at({1, 0}), 139.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 154.0f);
}

TEST(MatMulTest, RowTimesMatrix) {
    // [1 2 3] (1x3) * [1 0] (3x2) = [1*1+2*0+3*1  1*0+2*1+3*0] = [4 2]
    //                  [0 1]
    //                  [1 0]
    bmgpt::Tensor a({1, 3}, {1, 2, 3});
    bmgpt::Tensor b({3, 2}, {1, 0, 0, 1, 1, 0});
    bmgpt::Tensor c = bmgpt::matmul(a, b);

    EXPECT_EQ(c.shape(), (std::vector<size_t>{1, 2}));
    EXPECT_FLOAT_EQ(c.at({0, 0}), 4.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 2.0f);
}

TEST(MatMulTest, IdentityMatrix) {
    bmgpt::Tensor a({2, 2}, {3, 7, 1, 4});
    bmgpt::Tensor eye({2, 2}, {1, 0, 0, 1});
    bmgpt::Tensor c = bmgpt::matmul(a, eye);

    for (size_t i = 0; i < a.numel(); ++i)
        EXPECT_FLOAT_EQ(c.data()[i], a.data()[i]);
}

TEST(MatMulTest, SingleElement) {
    // [3] * [4] = [12]  (1x1 matrices)
    bmgpt::Tensor a({1, 1}, {3.0f});
    bmgpt::Tensor b({1, 1}, {4.0f});
    bmgpt::Tensor c = bmgpt::matmul(a, b);

    EXPECT_EQ(c.shape(), (std::vector<size_t>{1, 1}));
    EXPECT_FLOAT_EQ(c.at({0, 0}), 12.0f);
}

// ── Error cases ─────────────────────────────────────────────────────

TEST(MatMulTest, InnerDimensionMismatchThrows) {
    bmgpt::Tensor a({2, 3});
    bmgpt::Tensor b({2, 3});  // b.size(0)=2 != a.size(1)=3
    EXPECT_THROW(bmgpt::matmul(a, b), std::invalid_argument);
}

TEST(MatMulTest, NonTwoDimensionalThrows) {
    bmgpt::Tensor a3d({2, 3, 4});
    bmgpt::Tensor b2d({4, 2});
    EXPECT_THROW(bmgpt::matmul(a3d, b2d), std::invalid_argument);

    bmgpt::Tensor a1d({5});
    EXPECT_THROW(bmgpt::matmul(a1d, b2d), std::invalid_argument);
}
