/// @file test_feedforward.cpp
/// @brief Numerical validation and unit tests for bmgpt::FeedForward (Phase 3 — Layers).
///
/// Strategy:
///   - Load micro-model weights via WeightLoader.
///   - Construct FeedForward instances for block 0 ("h.0.mlp") and block 1 ("h.1.mlp").
///   - Test intermediate steps:
///       c_fc(x)          == block_N_mlp_fc_out.npy
///       gelu(c_fc(x))    == block_N_mlp_gelu_out.npy
///   - Test full forward pass:
///       forward(ln2_out) == block_N_mlp_proj_out.npy
///   - Verify error handling and shapes.

#include "baremetalgpt/FeedForward.hpp"
#include "baremetalgpt/WeightLoader.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

#ifndef WEIGHTS_DIR
#  error "WEIGHTS_DIR must be defined via target_compile_definitions"
#endif
#ifndef DUMPS_DIR
#  error "DUMPS_DIR must be defined via target_compile_definitions"
#endif

#define XSTR(x) STR(x)
#define STR(x) #x

static const std::string kWeightsDir   = XSTR(WEIGHTS_DIR);
static const std::string kDumpsDir     = XSTR(DUMPS_DIR);
static const std::string kMicroWeights = kWeightsDir + "/micro_gpt2_weights.safetensors";

// ---------------------------------------------------------------------------
// Minimal NPY v1.0 loader
// ---------------------------------------------------------------------------

static std::vector<float> load_npy(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_npy: cannot open \"" + path + "\"");
    char magic[6]; f.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY")
        throw std::runtime_error("load_npy: bad magic");
    uint8_t major = 0, minor = 0;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);
    if (major != 1) throw std::runtime_error("load_npy: unsupported version");
    (void)minor;
    uint16_t hlen = 0;
    f.read(reinterpret_cast<char*>(&hlen), 2);
    std::string header(hlen, '\0');
    f.read(&header[0], hlen);
    if (header.find("<f4") == std::string::npos)
        throw std::runtime_error("load_npy: expected float32");

    std::vector<float> data;
    float val = 0.0f;
    while (f.read(reinterpret_cast<char*>(&val), sizeof(float)))
        data.push_back(val);
    return data;
}

static void compare_tensors(const bmgpt::Tensor& actual,
                            const std::vector<float>& expected,
                            float tol = 1e-4f)
{
    ASSERT_EQ(actual.numel(), expected.size());
    float max_err = 0.0f;
    for (size_t i = 0; i < actual.numel(); ++i) {
        float err = std::abs(actual.data()[i] - expected[i]);
        if (err > max_err) max_err = err;
        EXPECT_NEAR(actual.data()[i], expected[i], tol)
            << "Mismatch at index " << i << " (max_err so far: " << max_err << ")";
    }
}

// ---------------------------------------------------------------------------
// Structural & Dimension Tests
// ---------------------------------------------------------------------------

TEST(FeedForwardTest, DimensionsBlock0) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::FeedForward ffn(loader, "h.0.mlp");

    EXPECT_EQ(ffn.d_model(), 64u);
    EXPECT_EQ(ffn.d_ff(),    256u);

    EXPECT_EQ(ffn.c_fc().in_features(),   64u);
    EXPECT_EQ(ffn.c_fc().out_features(),  256u);
    EXPECT_EQ(ffn.c_proj().in_features(), 256u);
    EXPECT_EQ(ffn.c_proj().out_features(), 64u);
}

TEST(FeedForwardTest, DimensionsBlock1) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::FeedForward ffn(loader, "h.1.mlp");

    EXPECT_EQ(ffn.d_model(), 64u);
    EXPECT_EQ(ffn.d_ff(),    256u);
}

TEST(FeedForwardTest, OutputShapePreservation) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::FeedForward ffn(loader, "h.0.mlp");

    bmgpt::Tensor x_seq4({4, 64});
    bmgpt::Tensor out4 = ffn.forward(x_seq4);
    EXPECT_EQ(out4.ndim(), 2u);
    EXPECT_EQ(out4.size(0), 4u);
    EXPECT_EQ(out4.size(1), 64u);

    bmgpt::Tensor x_seq1({1, 64});
    bmgpt::Tensor out1 = ffn.forward(x_seq1);
    EXPECT_EQ(out1.ndim(), 2u);
    EXPECT_EQ(out1.size(0), 1u);
    EXPECT_EQ(out1.size(1), 64u);
}

// ---------------------------------------------------------------------------
// Error Handling Tests
// ---------------------------------------------------------------------------

TEST(FeedForwardTest, MissingKeyThrows) {
    bmgpt::WeightLoader loader(kMicroWeights);
    EXPECT_THROW(bmgpt::FeedForward(loader, "nonexistent.layer.mlp"), std::runtime_error);
}

TEST(FeedForwardTest, Non2DInputThrows) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::FeedForward ffn(loader, "h.0.mlp");

    bmgpt::Tensor x_1d({64});
    EXPECT_THROW(ffn.forward(x_1d), std::invalid_argument);

    bmgpt::Tensor x_3d({1, 4, 64});
    EXPECT_THROW(ffn.forward(x_3d), std::invalid_argument);
}

TEST(FeedForwardTest, FeatureMismatchThrows) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::FeedForward ffn(loader, "h.0.mlp");

    bmgpt::Tensor x_bad({4, 128});
    EXPECT_THROW(ffn.forward(x_bad), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Numerical Validation: Block 0
// ---------------------------------------------------------------------------

TEST(FeedForwardTest, Block0_StepByStepAndFullMatch) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::FeedForward ffn(loader, "h.0.mlp");

    // Load reference inputs and intermediates
    std::vector<float> ln2_data   = load_npy(kDumpsDir + "/block_0_ln2_out.npy");
    std::vector<float> fc_data    = load_npy(kDumpsDir + "/block_0_mlp_fc_out.npy");
    std::vector<float> gelu_data  = load_npy(kDumpsDir + "/block_0_mlp_gelu_out.npy");
    std::vector<float> proj_data  = load_npy(kDumpsDir + "/block_0_mlp_proj_out.npy");

    bmgpt::Tensor ln2({4, 64}, ln2_data);

    // Step 1: c_fc
    bmgpt::Tensor fc_out = ffn.c_fc().forward(ln2);
    compare_tensors(fc_out, fc_data, 1e-4f);

    // Step 2: GELU
    bmgpt::Tensor gelu_out = bmgpt::gelu(fc_out);
    compare_tensors(gelu_out, gelu_data, 1e-4f);

    // Step 3: full forward pass
    bmgpt::Tensor ffn_out = ffn.forward(ln2);
    compare_tensors(ffn_out, proj_data, 1e-4f);
}

// ---------------------------------------------------------------------------
// Numerical Validation: Block 1
// ---------------------------------------------------------------------------

TEST(FeedForwardTest, Block1_FullMatch) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::FeedForward ffn(loader, "h.1.mlp");

    std::vector<float> ln2_data  = load_npy(kDumpsDir + "/block_1_ln2_out.npy");
    std::vector<float> proj_data = load_npy(kDumpsDir + "/block_1_mlp_proj_out.npy");

    bmgpt::Tensor ln2({4, 64}, ln2_data);
    bmgpt::Tensor ffn_out = ffn.forward(ln2);

    compare_tensors(ffn_out, proj_data, 1e-4f);
}
