/// @file test_activations.cpp
/// @brief Unit tests for bmgpt::gelu (Phase 3 — Layers).
///
/// Tests cover:
///   1. Scalar GELU — known analytical values (tolerance 1e-6)
///   2. Scalar GELU — boundary behaviour (x=0, large |x|)
///   3. Tensor GELU — output shape preserved
///   4. Tensor GELU — element-wise correctness (hand-computed)
///   5. Mathematical properties — near-odd symmetry, monotonicity in |x|
///   6. Numerical validation against reference dumps:
///        block_0_mlp_gelu_out.npy  (input: block_0_mlp_fc_out.npy)
///        block_1_mlp_gelu_out.npy  (input: block_1_mlp_fc_out.npy)
///
/// Tolerance for numerical tests: max_abs_diff < 1e-4 (project standard).
/// Tolerance for scalar spot-checks: 1e-6 (direct float arithmetic).

#include "baremetalgpt/activations.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

#ifndef DUMPS_DIR
#  error "DUMPS_DIR must be defined via target_compile_definitions"
#endif

#define XSTR(x) STR(x)
#define STR(x) #x

static const std::string kDumpsDir = XSTR(DUMPS_DIR);

// ---------------------------------------------------------------------------
// Minimal NPY v1.0 loader (same helper as test_layer_norm.cpp)
// ---------------------------------------------------------------------------

static std::vector<float> load_npy(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("load_npy: cannot open \"" + path + "\"");
    }
    char magic[6];
    f.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") {
        throw std::runtime_error("load_npy: bad magic in \"" + path + "\"");
    }
    uint8_t major = 0, minor = 0;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);
    if (major != 1) {
        throw std::runtime_error("load_npy: unsupported version");
    }
    (void)minor;
    uint16_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, '\0');
    f.read(&header[0], header_len);
    if (header.find("<f4") == std::string::npos) {
        throw std::runtime_error("load_npy: expected float32 dtype");
    }
    auto shape_pos = header.find("'shape':");
    if (shape_pos == std::string::npos) shape_pos = header.find("\"shape\":");
    if (shape_pos == std::string::npos) {
        throw std::runtime_error("load_npy: missing shape");
    }
    auto po = header.find('(', shape_pos);
    auto pc = header.find(')', po);
    std::string shape_str = header.substr(po + 1, pc - po - 1);
    size_t numel = 1;
    {
        std::string tok;
        for (char c : shape_str + ",") {
            if (c == ',') {
                size_t s = tok.find_first_not_of(" \t");
                size_t e = tok.find_last_not_of(" \t");
                if (s != std::string::npos) {
                    tok = tok.substr(s, e - s + 1);
                    numel *= static_cast<size_t>(std::stoul(tok));
                }
                tok.clear();
            } else {
                tok += c;
            }
        }
    }
    std::vector<float> data(numel);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(numel * sizeof(float)));
    if (!f) {
        throw std::runtime_error("load_npy: truncated data in \"" + path + "\"");
    }
    return data;
}

// ---------------------------------------------------------------------------
// Helper: element-wise max absolute diff check
// ---------------------------------------------------------------------------

static void expect_near_all(const std::string& label,
                             const bmgpt::Tensor& result,
                             const std::vector<float>& expected,
                             float tol = 1e-4f)
{
    ASSERT_EQ(result.numel(), expected.size())
        << label << ": element count mismatch";

    float max_diff = 0.0f;
    size_t worst = 0;
    for (size_t i = 0; i < expected.size(); ++i) {
        float d = std::abs(result.data()[i] - expected[i]);
        if (d > max_diff) { max_diff = d; worst = i; }
    }
    EXPECT_LT(max_diff, tol)
        << label << ": max_abs_diff=" << max_diff
        << " at index " << worst
        << " (got=" << result.data()[worst]
        << " expected=" << expected[worst] << ")";
}

// ---------------------------------------------------------------------------
// 1. Scalar GELU — analytical spot-checks
//    Ground truth computed with Python (double precision):
//      gelu(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
// ---------------------------------------------------------------------------

TEST(GeluScalar, Zero) {
    // GELU(0) = 0 exactly (tanh(0)=0, so 0.5*0*(...) = 0)
    EXPECT_NEAR(bmgpt::gelu(0.0f), 0.0f, 1e-7f);
}

TEST(GeluScalar, PositiveOne) {
    // gelu(1.0) ≈ 0.8411919906
    EXPECT_NEAR(bmgpt::gelu(1.0f), 0.8411919906f, 1e-6f);
}

TEST(GeluScalar, NegativeOne) {
    // gelu(-1.0) ≈ -0.1588080094
    EXPECT_NEAR(bmgpt::gelu(-1.0f), -0.1588080094f, 1e-6f);
}

TEST(GeluScalar, PositiveTwo) {
    // gelu(2.0) ≈ 1.9545976941
    EXPECT_NEAR(bmgpt::gelu(2.0f), 1.9545976941f, 1e-5f);
}

TEST(GeluScalar, NegativeTwo) {
    // gelu(-2.0) ≈ -0.0454023059
    EXPECT_NEAR(bmgpt::gelu(-2.0f), -0.0454023059f, 1e-6f);
}

TEST(GeluScalar, HalfPositive) {
    // gelu(0.5) ≈ 0.3457140098
    EXPECT_NEAR(bmgpt::gelu(0.5f), 0.3457140098f, 1e-6f);
}

TEST(GeluScalar, HalfNegative) {
    // gelu(-0.5) ≈ -0.1542859902
    EXPECT_NEAR(bmgpt::gelu(-0.5f), -0.1542859902f, 1e-6f);
}

// ---------------------------------------------------------------------------
// 2. Scalar GELU — saturation at large |x|
// ---------------------------------------------------------------------------

TEST(GeluScalar, LargePositiveSaturatesToX) {
    // For very large x, tanh→1, so GELU(x)→x
    EXPECT_NEAR(bmgpt::gelu(100.0f), 100.0f, 1e-3f);
}

TEST(GeluScalar, LargeNegativeSaturatesToZero) {
    // For very large |x|<0, tanh→-1, so GELU(x)→0
    EXPECT_NEAR(bmgpt::gelu(-100.0f), 0.0f, 1e-3f);
}

// ---------------------------------------------------------------------------
// 3. Mathematical properties
// ---------------------------------------------------------------------------

TEST(GeluScalar, MonotonicForPositiveInputs) {
    // GELU should be strictly increasing for x > 0
    float prev = bmgpt::gelu(0.0f);
    for (int i = 1; i <= 10; ++i) {
        float cur = bmgpt::gelu(static_cast<float>(i) * 0.5f);
        EXPECT_GT(cur, prev) << "Not monotonic at x=" << i * 0.5f;
        prev = cur;
    }
}

TEST(GeluScalar, SignPreservation) {
    // GELU(x) has the same sign as x for all x:
    //   x > 0  =>  GELU(x) > 0  (0.5*x*(1+tanh(...)) with tanh∈(-1,1))
    //   x < 0  =>  GELU(x) < 0  (tanh always < 1, so 1+tanh < 2, factor < x*0 ... actually
    //              gelu(-x) < 0 because 0.5*(-x)*(1+tanh(positive)) < 0 when x>0)
    //   x = 0  =>  GELU(0) = 0
    for (float x : {0.1f, 0.5f, 1.0f, 2.0f, 5.0f}) {
        EXPECT_GT(bmgpt::gelu( x),  0.0f) << "Sign wrong for x= " <<  x;
        EXPECT_LT(bmgpt::gelu(-x),  0.0f) << "Sign wrong for x=-" <<  x;
    }
    EXPECT_NEAR(bmgpt::gelu(0.0f), 0.0f, 1e-7f);
}

// ---------------------------------------------------------------------------
// 4. Tensor overload — shape preservation
// ---------------------------------------------------------------------------

TEST(GeluTensor, OutputShapeMatchesInput_2D) {
    bmgpt::Tensor x({4, 256});
    auto out = bmgpt::gelu(x);
    ASSERT_EQ(out.ndim(), 2u);
    EXPECT_EQ(out.size(0), 4u);
    EXPECT_EQ(out.size(1), 256u);
}

TEST(GeluTensor, OutputShapeMatchesInput_1D) {
    bmgpt::Tensor x({64});
    auto out = bmgpt::gelu(x);
    ASSERT_EQ(out.ndim(), 1u);
    EXPECT_EQ(out.size(0), 64u);
}

TEST(GeluTensor, OutputShapeMatchesInput_3D) {
    bmgpt::Tensor x({2, 4, 32});
    auto out = bmgpt::gelu(x);
    ASSERT_EQ(out.ndim(), 3u);
    EXPECT_EQ(out.size(0), 2u);
    EXPECT_EQ(out.size(1), 4u);
    EXPECT_EQ(out.size(2), 32u);
}

// ---------------------------------------------------------------------------
// 5. Tensor overload — element-wise correctness
// ---------------------------------------------------------------------------

TEST(GeluTensor, ElementWiseCorrectness) {
    // Build a small [2, 3] tensor with known values
    std::vector<float> vals = {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f};
    bmgpt::Tensor x({2, 3}, vals);

    auto out = bmgpt::gelu(x);
    ASSERT_EQ(out.numel(), 6u);

    const float expected[] = {
        0.0f,           // gelu(0)
        0.8411919906f,  // gelu(1)
       -0.1588080094f,  // gelu(-1)
        0.3457140098f,  // gelu(0.5)
       -0.1542859902f,  // gelu(-0.5)
        1.9545976941f,  // gelu(2)
    };

    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(out.data()[i], expected[i], 1e-5f)
            << "at index " << i;
    }
}

TEST(GeluTensor, InputNotModified) {
    std::vector<float> vals = {1.0f, -1.0f, 0.5f};
    bmgpt::Tensor x({3}, vals);
    auto out = bmgpt::gelu(x);
    // Original tensor must be unchanged
    EXPECT_NEAR(x.data()[0],  1.0f, 1e-7f);
    EXPECT_NEAR(x.data()[1], -1.0f, 1e-7f);
    EXPECT_NEAR(x.data()[2],  0.5f, 1e-7f);
    (void)out;
}

// ---------------------------------------------------------------------------
// 6. Numerical validation — block 0 MLP GELU
//    Input:  block_0_mlp_fc_out.npy  [4, 256]
//    Output: block_0_mlp_gelu_out.npy [4, 256]
// ---------------------------------------------------------------------------

TEST(GeluNumerical, Block0MlpGelu_NumericalMatch) {
    auto fc_data   = load_npy(kDumpsDir + "/block_0_mlp_fc_out.npy");
    auto ref_data  = load_npy(kDumpsDir + "/block_0_mlp_gelu_out.npy");

    bmgpt::Tensor fc_out({4, 256}, fc_data);
    auto result = bmgpt::gelu(fc_out);

    expect_near_all("block_0_gelu", result, ref_data);
}

// ---------------------------------------------------------------------------
// 7. Numerical validation — block 1 MLP GELU
//    Input:  block_1_mlp_fc_out.npy  [4, 256]
//    Output: block_1_mlp_gelu_out.npy [4, 256]
// ---------------------------------------------------------------------------

TEST(GeluNumerical, Block1MlpGelu_NumericalMatch) {
    auto fc_data   = load_npy(kDumpsDir + "/block_1_mlp_fc_out.npy");
    auto ref_data  = load_npy(kDumpsDir + "/block_1_mlp_gelu_out.npy");

    bmgpt::Tensor fc_out({4, 256}, fc_data);
    auto result = bmgpt::gelu(fc_out);

    expect_near_all("block_1_gelu", result, ref_data);
}
