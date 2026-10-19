/// @file test_causal_self_attention.cpp
/// @brief Numerical validation and unit tests for bmgpt::CausalSelfAttention (Phase 3 — Layers).
///
/// Strategy:
///   - Load micro-model weights via WeightLoader.
///   - Verify architecture dimensions and error handling.
///   - Validate the fundamental causality invariant:
///       output[0] for sequence [x0, x1, ...] MUST match output[0] for [x0] alone.
///   - Compare full forward pass outputs on block 0 and block 1 against
///     the PyTorch reference dumps (block_0_attn_proj.npy & block_1_attn_proj.npy).

#include "baremetalgpt/CausalSelfAttention.hpp"
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

TEST(CausalSelfAttentionTest, DimensionsBlock0) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::CausalSelfAttention attn(loader, "h.0.attn", 2);

    EXPECT_EQ(attn.n_heads(),  2u);
    EXPECT_EQ(attn.d_model(),  64u);
    EXPECT_EQ(attn.head_dim(), 32u);

    EXPECT_EQ(attn.c_attn().in_features(),  64u);
    EXPECT_EQ(attn.c_attn().out_features(), 192u);
    EXPECT_EQ(attn.c_proj().in_features(),  64u);
    EXPECT_EQ(attn.c_proj().out_features(), 64u);
}

TEST(CausalSelfAttentionTest, DimensionsBlock1) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::CausalSelfAttention attn(loader, "h.1.attn", 2);

    EXPECT_EQ(attn.n_heads(),  2u);
    EXPECT_EQ(attn.d_model(),  64u);
    EXPECT_EQ(attn.head_dim(), 32u);
}

TEST(CausalSelfAttentionTest, OutputShapePreservation) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::CausalSelfAttention attn(loader, "h.0.attn", 2);

    bmgpt::Tensor x_seq4({4, 64});
    bmgpt::Tensor out4 = attn.forward(x_seq4);
    EXPECT_EQ(out4.ndim(), 2u);
    EXPECT_EQ(out4.size(0), 4u);
    EXPECT_EQ(out4.size(1), 64u);

    bmgpt::Tensor x_seq1({1, 64});
    bmgpt::Tensor out1 = attn.forward(x_seq1);
    EXPECT_EQ(out1.ndim(), 2u);
    EXPECT_EQ(out1.size(0), 1u);
    EXPECT_EQ(out1.size(1), 64u);
}

// ---------------------------------------------------------------------------
// Error Handling Tests
// ---------------------------------------------------------------------------

TEST(CausalSelfAttentionTest, ZeroHeadsThrows) {
    bmgpt::WeightLoader loader(kMicroWeights);
    EXPECT_THROW(bmgpt::CausalSelfAttention(loader, "h.0.attn", 0), std::runtime_error);
}

TEST(CausalSelfAttentionTest, IndivisibleHeadsThrows) {
    bmgpt::WeightLoader loader(kMicroWeights);
    // 64 is not divisible by 5
    EXPECT_THROW(bmgpt::CausalSelfAttention(loader, "h.0.attn", 5), std::runtime_error);
}

TEST(CausalSelfAttentionTest, MissingKeyThrows) {
    bmgpt::WeightLoader loader(kMicroWeights);
    EXPECT_THROW(bmgpt::CausalSelfAttention(loader, "unknown.attn", 2), std::runtime_error);
}

TEST(CausalSelfAttentionTest, Non2DInputThrows) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::CausalSelfAttention attn(loader, "h.0.attn", 2);

    bmgpt::Tensor x_1d({64});
    EXPECT_THROW(attn.forward(x_1d), std::invalid_argument);

    bmgpt::Tensor x_3d({1, 4, 64});
    EXPECT_THROW(attn.forward(x_3d), std::invalid_argument);
}

TEST(CausalSelfAttentionTest, FeatureMismatchThrows) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::CausalSelfAttention attn(loader, "h.0.attn", 2);

    bmgpt::Tensor x_bad({4, 32});
    EXPECT_THROW(attn.forward(x_bad), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Causality Invariant Test
// ---------------------------------------------------------------------------

TEST(CausalSelfAttentionTest, CausalityInvariant) {
    // In causal attention, future tokens cannot influence earlier tokens.
    // Therefore, output for token 0 must be identical whether the sequence
    // has length 1 or length 4 (with subsequent tokens arbitrarily modified).
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::CausalSelfAttention attn(loader, "h.0.attn", 2);

    std::vector<float> ln1_data = load_npy(kDumpsDir + "/block_0_ln1_out.npy"); // 4 tokens * 64
    bmgpt::Tensor x_full({4, 64}, ln1_data);

    // Extract first token alone [1, 64]
    std::vector<float> token0_data(ln1_data.begin(), ln1_data.begin() + 64);
    bmgpt::Tensor x_single({1, 64}, token0_data);

    bmgpt::Tensor out_full   = attn.forward(x_full);
    bmgpt::Tensor out_single = attn.forward(x_single);

    // Verify row 0 matches exactly
    for (size_t d = 0; d < 64; ++d) {
        EXPECT_NEAR(out_full.data()[d], out_single.data()[d], 1e-6f)
            << "Causality violated at feature " << d;
    }
}

// ---------------------------------------------------------------------------
// Numerical Validation: Block 0 & Block 1
// ---------------------------------------------------------------------------

TEST(CausalSelfAttentionTest, Block0_NumericalMatch) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::CausalSelfAttention attn(loader, "h.0.attn", 2);

    std::vector<float> ln1_data  = load_npy(kDumpsDir + "/block_0_ln1_out.npy");
    std::vector<float> proj_data = load_npy(kDumpsDir + "/block_0_attn_proj.npy");

    bmgpt::Tensor ln1({4, 64}, ln1_data);
    bmgpt::Tensor out = attn.forward(ln1);

    compare_tensors(out, proj_data, 1e-4f);
}

TEST(CausalSelfAttentionTest, Block1_NumericalMatch) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::CausalSelfAttention attn(loader, "h.1.attn", 2);

    std::vector<float> ln1_data  = load_npy(kDumpsDir + "/block_1_ln1_out.npy");
    std::vector<float> proj_data = load_npy(kDumpsDir + "/block_1_attn_proj.npy");

    bmgpt::Tensor ln1({4, 64}, ln1_data);
    bmgpt::Tensor out = attn.forward(ln1);

    compare_tensors(out, proj_data, 1e-4f);
}
