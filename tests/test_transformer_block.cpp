/// @file test_transformer_block.cpp
/// @brief Unit tests and numerical validation for bmgpt::TransformerBlock (Phase 4).
///
/// Strategy:
///   - Load micro-model weights via WeightLoader (n_layers=2, n_heads=2,
///     d_model=64, d_ff=256, context_length=64).
///   - Verify structural properties of the constructed block.
///   - Verify error-handling paths (non-2D input, wrong d_model).
///   - Run forward() on both blocks and compare against reference dumps:
///       block_0_resid1.npy  == x + attn(ln_1(x))  for block 0
///       block_0_out.npy     == full block 0 output (after MLP residual)
///       block_1_resid1.npy  == x + attn(ln_1(x))  for block 1
///       block_1_out.npy     == full block 1 output
///   - Tolerance: 1e-4 (matches Phase 4 spec; floating-point accumulation
///     across matmul + softmax is larger than pure lookup tests).

#include "baremetalgpt/TransformerBlock.hpp"
#include "baremetalgpt/Embedding.hpp"
#include "baremetalgpt/Tensor.hpp"
#include "baremetalgpt/WeightLoader.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Path helpers (injected by CMake — see tests/CMakeLists.txt)
// ---------------------------------------------------------------------------

#ifndef WEIGHTS_DIR
#  error "WEIGHTS_DIR must be defined via target_compile_definitions"
#endif
#ifndef DUMPS_DIR
#  error "DUMPS_DIR must be defined via target_compile_definitions"
#endif

#define XSTR(x) STR(x)
#define STR(x)  #x

static const std::string kWeightsDir   = XSTR(WEIGHTS_DIR);
static const std::string kDumpsDir     = XSTR(DUMPS_DIR);
static const std::string kMicroWeights = kWeightsDir + "/micro_gpt2_weights.safetensors";

// Micro-model hyperparameters (must match metadata.json)
static constexpr size_t kNHeads  = 2;
static constexpr size_t kDModel  = 64;

// Canonical test sequence (must match dump_reference.py MICRO_INPUT_IDS)
static const std::vector<int32_t> kInputIds = {42, 105, 12, 88};

// ---------------------------------------------------------------------------
// Minimal NPY v1.0 loader (shared pattern from Phase 3/4 tests)
// ---------------------------------------------------------------------------

static std::vector<float> load_npy(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_npy: cannot open \"" + path + "\"");
    char magic[6]; f.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY")
        throw std::runtime_error("load_npy: bad magic in " + path);
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
        throw std::runtime_error("load_npy: expected float32 in " + path);

    std::vector<float> data;
    float val = 0.0f;
    while (f.read(reinterpret_cast<char*>(&val), sizeof(float)))
        data.push_back(val);
    return data;
}

// ---------------------------------------------------------------------------
// Helper: load initial hidden state h0 = Embedding.forward(kInputIds)
// ---------------------------------------------------------------------------

static bmgpt::Tensor load_initial_hidden()
{
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Embedding    emb(loader);
    return emb.forward(kInputIds);
}

// ---------------------------------------------------------------------------
// Helper: compare Tensor against reference vector
// ---------------------------------------------------------------------------

static void compare_tensors(const bmgpt::Tensor&      actual,
                             const std::vector<float>& expected,
                             float                     tol = 1e-4f)
{
    ASSERT_EQ(actual.numel(), expected.size())
        << "numel mismatch: C++ has " << actual.numel()
        << ", reference has " << expected.size();

    float max_err = 0.0f;
    for (size_t i = 0; i < actual.numel(); ++i) {
        float err = std::abs(actual.data()[i] - expected[i]);
        if (err > max_err) max_err = err;
        EXPECT_NEAR(actual.data()[i], expected[i], tol)
            << "Mismatch at flat index " << i
            << " (max_err so far: " << max_err << ")";
    }
}

// ---------------------------------------------------------------------------
// Structural Tests
// ---------------------------------------------------------------------------

TEST(TransformerBlockTest, ConstructBlock0FromWeightLoader) {
    bmgpt::WeightLoader loader(kMicroWeights);
    EXPECT_NO_THROW(bmgpt::TransformerBlock blk(loader, "h.0", kNHeads));
}

TEST(TransformerBlockTest, ConstructBlock1FromWeightLoader) {
    bmgpt::WeightLoader loader(kMicroWeights);
    EXPECT_NO_THROW(bmgpt::TransformerBlock blk(loader, "h.1", kNHeads));
}

TEST(TransformerBlockTest, DModelIsCorrect) {
    bmgpt::WeightLoader    loader(kMicroWeights);
    bmgpt::TransformerBlock blk(loader, "h.0", kNHeads);

    EXPECT_EQ(blk.d_model(), kDModel);
}

TEST(TransformerBlockTest, NHeadsIsCorrect) {
    bmgpt::WeightLoader    loader(kMicroWeights);
    bmgpt::TransformerBlock blk(loader, "h.0", kNHeads);

    EXPECT_EQ(blk.n_heads(), kNHeads);
}

TEST(TransformerBlockTest, OutputShapePreserved) {
    bmgpt::WeightLoader    loader(kMicroWeights);
    bmgpt::TransformerBlock blk(loader, "h.0", kNHeads);

    bmgpt::Tensor h0  = load_initial_hidden();
    bmgpt::Tensor out = blk.forward(h0);

    // Shape must be preserved: [T, d_model]
    EXPECT_EQ(out.ndim(),   2u);
    EXPECT_EQ(out.size(0),  h0.size(0));   // T = 4
    EXPECT_EQ(out.size(1),  kDModel);
}

TEST(TransformerBlockTest, OutputShapeSingleToken) {
    bmgpt::WeightLoader    loader(kMicroWeights);
    bmgpt::TransformerBlock blk(loader, "h.0", kNHeads);

    // Build a single-token input manually from initial hidden
    // (just use embedding of token 0)
    bmgpt::Embedding emb(loader);
    bmgpt::Tensor    h_single = emb.forward({0});

    bmgpt::Tensor out = blk.forward(h_single);
    EXPECT_EQ(out.ndim(),  2u);
    EXPECT_EQ(out.size(0), 1u);
    EXPECT_EQ(out.size(1), kDModel);
}

// ---------------------------------------------------------------------------
// Error Handling Tests
// ---------------------------------------------------------------------------

TEST(TransformerBlockTest, Non2DInputThrows) {
    bmgpt::WeightLoader    loader(kMicroWeights);
    bmgpt::TransformerBlock blk(loader, "h.0", kNHeads);

    // 1-D tensor
    bmgpt::Tensor bad1d({kDModel});
    EXPECT_THROW(blk.forward(bad1d), std::invalid_argument);

    // 3-D tensor
    bmgpt::Tensor bad3d({4, kDModel, 1});
    EXPECT_THROW(blk.forward(bad3d), std::invalid_argument);
}

TEST(TransformerBlockTest, WrongDModelThrows) {
    bmgpt::WeightLoader    loader(kMicroWeights);
    bmgpt::TransformerBlock blk(loader, "h.0", kNHeads);

    // d_model=32 instead of 64
    bmgpt::Tensor bad({4, 32u});
    EXPECT_THROW(blk.forward(bad), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Numerical Validation — Block 0
// ---------------------------------------------------------------------------

TEST(TransformerBlockNumerical, Block0_Resid1_NumericalMatch) {
    // Verify x + attn(ln_1(x)) == block_0_resid1.npy
    bmgpt::WeightLoader    loader(kMicroWeights);
    bmgpt::TransformerBlock blk0(loader, "h.0", kNHeads);

    bmgpt::Tensor h0 = load_initial_hidden();

    // We need to replicate the first residual step only.
    // The reference dump block_0_resid1.npy stores exactly x1 = x + attn(ln_1(x)).
    // We cannot call only half of forward(), so we verify the intermediate via
    // manually reconstructing it from the sub-layers.
    const bmgpt::Tensor ln1_out  = blk0.ln_1().forward(h0);
    const bmgpt::Tensor attn_out = blk0.attn().forward(ln1_out);

    const size_t T      = h0.size(0);
    const size_t dm     = h0.size(1);
    const size_t numel  = T * dm;
    bmgpt::Tensor resid1({T, dm});
    for (size_t i = 0; i < numel; ++i) {
        resid1.data()[i] = h0.data()[i] + attn_out.data()[i];
    }

    std::vector<float> ref = load_npy(kDumpsDir + "/block_0_resid1.npy");
    compare_tensors(resid1, ref, 1e-4f);
}

TEST(TransformerBlockNumerical, Block0_Out_NumericalMatch) {
    // Verify full block 0 output == block_0_out.npy
    bmgpt::WeightLoader    loader(kMicroWeights);
    bmgpt::TransformerBlock blk0(loader, "h.0", kNHeads);

    bmgpt::Tensor h0  = load_initial_hidden();
    bmgpt::Tensor out = blk0.forward(h0);

    std::vector<float> ref = load_npy(kDumpsDir + "/block_0_out.npy");
    compare_tensors(out, ref, 1e-4f);
}

// ---------------------------------------------------------------------------
// Numerical Validation — Block 1
// ---------------------------------------------------------------------------

TEST(TransformerBlockNumerical, Block1_Resid1_NumericalMatch) {
    // Block 1 input == block 0 output; verify x + attn(ln_1(x)) for block 1.
    bmgpt::WeightLoader    loader(kMicroWeights);
    bmgpt::TransformerBlock blk0(loader, "h.0", kNHeads);
    bmgpt::TransformerBlock blk1(loader, "h.1", kNHeads);

    bmgpt::Tensor h0 = load_initial_hidden();
    bmgpt::Tensor h1 = blk0.forward(h0);  // block 1 input = block 0 output

    const bmgpt::Tensor ln1_out  = blk1.ln_1().forward(h1);
    const bmgpt::Tensor attn_out = blk1.attn().forward(ln1_out);

    const size_t T      = h1.size(0);
    const size_t dm     = h1.size(1);
    const size_t numel  = T * dm;
    bmgpt::Tensor resid1({T, dm});
    for (size_t i = 0; i < numel; ++i) {
        resid1.data()[i] = h1.data()[i] + attn_out.data()[i];
    }

    std::vector<float> ref = load_npy(kDumpsDir + "/block_1_resid1.npy");
    compare_tensors(resid1, ref, 1e-4f);
}

TEST(TransformerBlockNumerical, Block1_Out_NumericalMatch) {
    // Verify full block 1 output == block_1_out.npy
    bmgpt::WeightLoader    loader(kMicroWeights);
    bmgpt::TransformerBlock blk0(loader, "h.0", kNHeads);
    bmgpt::TransformerBlock blk1(loader, "h.1", kNHeads);

    bmgpt::Tensor h0  = load_initial_hidden();
    bmgpt::Tensor h1  = blk0.forward(h0);
    bmgpt::Tensor out = blk1.forward(h1);

    std::vector<float> ref = load_npy(kDumpsDir + "/block_1_out.npy");
    compare_tensors(out, ref, 1e-4f);
}

