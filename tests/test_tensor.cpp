#include <gtest/gtest.h>
#include "baremetalgpt/Tensor.hpp"

// ── Constructors ────────────────────────────────────────────────────

TEST(TensorTest, DefaultConstructible) {
    bmgpt::Tensor t;
    EXPECT_EQ(t.ndim(), 0u);
    EXPECT_EQ(t.numel(), 0u);
}

TEST(TensorTest, ShapeConstructor) {
    bmgpt::Tensor t({2, 3});
    EXPECT_EQ(t.ndim(), 2u);
    EXPECT_EQ(t.numel(), 6u);
    EXPECT_EQ(t.size(0), 2u);
    EXPECT_EQ(t.size(1), 3u);
    // Strides should be row-major: {3, 1}
    EXPECT_EQ(t.strides()[0], 3u);
    EXPECT_EQ(t.strides()[1], 1u);
}

TEST(TensorTest, ShapeConstructorZeroInitialized) {
    bmgpt::Tensor t({2, 3});
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_FLOAT_EQ(t.at({i, j}), 0.0f);
}

TEST(TensorTest, DataConstructor) {
    bmgpt::Tensor t({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    EXPECT_EQ(t.numel(), 4u);
    EXPECT_FLOAT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(t.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(t.at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(t.at({1, 1}), 4.0f);
}

TEST(TensorTest, DataConstructorSizeMismatchThrows) {
    EXPECT_THROW(
        bmgpt::Tensor({2, 2}, {1.0f, 2.0f, 3.0f}),
        std::invalid_argument
    );
    EXPECT_THROW(
        bmgpt::Tensor({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f}),
        std::invalid_argument
    );
}

TEST(TensorTest, OneDimensional) {
    bmgpt::Tensor t({5}, {10.0f, 20.0f, 30.0f, 40.0f, 50.0f});
    EXPECT_EQ(t.ndim(), 1u);
    EXPECT_EQ(t.numel(), 5u);
    EXPECT_FLOAT_EQ(t.at({2}), 30.0f);
}

// ── 3D Tensor ───────────────────────────────────────────────────────

TEST(TensorTest, ThreeDimensional) {
    bmgpt::Tensor t({2, 3, 4});
    EXPECT_EQ(t.ndim(), 3u);
    EXPECT_EQ(t.numel(), 24u);
    EXPECT_EQ(t.strides()[0], 12u);
    EXPECT_EQ(t.strides()[1], 4u);
    EXPECT_EQ(t.strides()[2], 1u);

    t.at({1, 2, 3}) = 42.0f;
    EXPECT_FLOAT_EQ(t.at({1, 2, 3}), 42.0f);
    EXPECT_FLOAT_EQ(t.data()[23], 42.0f);
}

// ── Bounds checking ─────────────────────────────────────────────────

TEST(TensorTest, AtWrongNumberOfIndicesThrows) {
    bmgpt::Tensor t({2, 3});
    EXPECT_THROW(t.at({0}), std::out_of_range);
    EXPECT_THROW(t.at({0, 0, 0}), std::out_of_range);
}

TEST(TensorTest, AtOutOfBoundsThrows) {
    bmgpt::Tensor t({2, 3});
    EXPECT_THROW(t.at({2, 0}), std::out_of_range);
    EXPECT_THROW(t.at({0, 3}), std::out_of_range);
}

TEST(TensorTest, SizeOutOfRangeThrows) {
    bmgpt::Tensor t({2, 3});
    EXPECT_THROW(t.size(2), std::out_of_range);
    EXPECT_THROW(t.size(100), std::out_of_range);
}

// ── Arithmetic ──────────────────────────────────────────────────────

TEST(TensorTest, AdditionSameShape) {
    bmgpt::Tensor a({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    bmgpt::Tensor b({2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});
    bmgpt::Tensor c = a + b;

    EXPECT_EQ(c.shape(), a.shape());
    EXPECT_FLOAT_EQ(c.at({0, 0}), 11.0f);
    EXPECT_FLOAT_EQ(c.at({0, 1}), 22.0f);
    EXPECT_FLOAT_EQ(c.at({1, 0}), 33.0f);
    EXPECT_FLOAT_EQ(c.at({1, 1}), 44.0f);
}

TEST(TensorTest, AdditionShapeMismatchThrows) {
    bmgpt::Tensor a({2, 3});
    bmgpt::Tensor b({3, 2});
    EXPECT_THROW(a + b, std::invalid_argument);
}

// ── Reshape ─────────────────────────────────────────────────────────

TEST(TensorTest, ReshapeBasic) {
    bmgpt::Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    bmgpt::Tensor r = t.reshape({3, 2});

    EXPECT_EQ(r.shape(), (std::vector<size_t>{3, 2}));
    EXPECT_EQ(r.numel(), 6u);
    // Row-major data order is preserved:
    // flat [1,2,3,4,5,6] → [3,2] gives rows [1,2], [3,4], [5,6]
    EXPECT_FLOAT_EQ(r.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(r.at({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(r.at({1, 0}), 3.0f);
    EXPECT_FLOAT_EQ(r.at({1, 1}), 4.0f);
    EXPECT_FLOAT_EQ(r.at({2, 0}), 5.0f);
    EXPECT_FLOAT_EQ(r.at({2, 1}), 6.0f);
}

TEST(TensorTest, ReshapeTo1D) {
    bmgpt::Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    bmgpt::Tensor r = t.reshape({6});
    EXPECT_EQ(r.ndim(), 1u);
    EXPECT_EQ(r.numel(), 6u);
    for (size_t i = 0; i < 6; ++i)
        EXPECT_FLOAT_EQ(r.at({i}), static_cast<float>(i + 1));
}

TEST(TensorTest, ReshapeTo3D) {
    bmgpt::Tensor t({6}, {1, 2, 3, 4, 5, 6});
    bmgpt::Tensor r = t.reshape({1, 2, 3});
    EXPECT_EQ(r.shape(), (std::vector<size_t>{1, 2, 3}));
    EXPECT_FLOAT_EQ(r.at({0, 0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(r.at({0, 1, 2}), 6.0f);
}

TEST(TensorTest, ReshapeSizeMismatchThrows) {
    bmgpt::Tensor t({2, 3});
    EXPECT_THROW(t.reshape({2, 4}), std::invalid_argument);
    EXPECT_THROW(t.reshape({7}), std::invalid_argument);
}

// ── Transpose ───────────────────────────────────────────────────────

TEST(TensorTest, Transpose2D) {
    // [1 2 3]^T = [1 4]
    // [4 5 6]     [2 5]
    //             [3 6]
    bmgpt::Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    bmgpt::Tensor r = t.transpose(0, 1);

    EXPECT_EQ(r.shape(), (std::vector<size_t>{3, 2}));
    EXPECT_FLOAT_EQ(r.at({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(r.at({0, 1}), 4.0f);
    EXPECT_FLOAT_EQ(r.at({1, 0}), 2.0f);
    EXPECT_FLOAT_EQ(r.at({1, 1}), 5.0f);
    EXPECT_FLOAT_EQ(r.at({2, 0}), 3.0f);
    EXPECT_FLOAT_EQ(r.at({2, 1}), 6.0f);
}

TEST(TensorTest, Transpose3DSwap01) {
    // Shape {2, 3, 4}, swap dims 0 and 1 → {3, 2, 4}
    // Verify: t[i][j][k] == r[j][i][k]
    bmgpt::Tensor t({2, 3, 4});
    for (size_t i = 0; i < 24; ++i)
        t.data()[i] = static_cast<float>(i);

    bmgpt::Tensor r = t.transpose(0, 1);
    EXPECT_EQ(r.shape(), (std::vector<size_t>{3, 2, 4}));

    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
                EXPECT_FLOAT_EQ(t.at({i, j, k}), r.at({j, i, k}));
}

TEST(TensorTest, Transpose3DSwap12) {
    // Shape {2, 3, 4}, swap dims 1 and 2 → {2, 4, 3}
    // This is the K-transpose needed in attention: K[n_heads, d_head, seq]
    bmgpt::Tensor t({2, 3, 4});
    for (size_t i = 0; i < 24; ++i)
        t.data()[i] = static_cast<float>(i);

    bmgpt::Tensor r = t.transpose(1, 2);
    EXPECT_EQ(r.shape(), (std::vector<size_t>{2, 4, 3}));

    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
                EXPECT_FLOAT_EQ(t.at({i, j, k}), r.at({i, k, j}));
}

TEST(TensorTest, Transpose3DSwap02) {
    // Shape {2, 3, 4}, swap dims 0 and 2 → {4, 3, 2}
    // The hardest case: both extremity dimensions swap, strides differ maximally.
    // Verify: t[i][j][k] == r[k][j][i]
    bmgpt::Tensor t({2, 3, 4});
    for (size_t i = 0; i < 24; ++i)
        t.data()[i] = static_cast<float>(i);

    bmgpt::Tensor r = t.transpose(0, 2);
    EXPECT_EQ(r.shape(), (std::vector<size_t>{4, 3, 2}));

    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            for (size_t k = 0; k < 4; ++k)
                EXPECT_FLOAT_EQ(t.at({i, j, k}), r.at({k, j, i}));
}

TEST(TensorTest, TransposeSameDimIsNoOp) {
    bmgpt::Tensor t({2, 3}, {1, 2, 3, 4, 5, 6});
    bmgpt::Tensor r = t.transpose(0, 0);
    EXPECT_EQ(r.shape(), t.shape());
    for (size_t i = 0; i < t.numel(); ++i)
        EXPECT_FLOAT_EQ(r.data()[i], t.data()[i]);
}

TEST(TensorTest, TransposeOutOfBoundsThrows) {
    bmgpt::Tensor t({2, 3});
    EXPECT_THROW(t.transpose(0, 2), std::out_of_range);
    EXPECT_THROW(t.transpose(3, 0), std::out_of_range);
}

// ── Debug utilities ─────────────────────────────────────────────────

TEST(TensorTest, ShapeToString) {
    EXPECT_EQ(bmgpt::Tensor({3, 4, 5}).shape_to_string(), "[3, 4, 5]");
    EXPECT_EQ(bmgpt::Tensor({1}).shape_to_string(), "[1]");
    EXPECT_EQ(bmgpt::Tensor().shape_to_string(), "[]");
}
