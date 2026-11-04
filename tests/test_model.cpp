/// @file test_model.cpp
/// @brief Unit tests and numerical validation for bmgpt::Model (Phase 4 — Tâche 3).
///
/// Strategy:
///   - Instantiate Model from micro_gpt2_weights.safetensors via a ModelConfig
///     matching metadata.json (n_layers=2, n_heads=2, d_model=64, vocab_size=256,
///     context_length=64, d_ff=256).
///   - Verify structural properties (config, sub-layer counts).
///   - Verify output shapes for T=1 and T=4.
///   - Validate the final layer-norm output against final_ln_out.npy (< 1e-4).
///   - Validate full logits against logits.npy (< 1e-4) on [42, 105, 12, 88].
///   - Verify argmax of the last-token logits matches the reference prediction.

#include "baremetalgpt/Model.hpp"
#include "baremetalgpt/WeightLoader.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Path helpers (injected by CMake)
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

// ---------------------------------------------------------------------------
// Micro-model config (must match reference_dumps/micro/metadata.json)
// ---------------------------------------------------------------------------

static bmgpt::ModelConfig micro_config()
{
    bmgpt::ModelConfig cfg;
    cfg.n_layers       = 2;
    cfg.n_heads        = 2;
    cfg.d_model        = 64;
    cfg.vocab_size     = 256;
    cfg.context_length = 64;
    cfg.d_ff           = 256;
    cfg.eps            = 1e-5f;
    return cfg;
}

// Canonical test sequence (matches dump_reference.py MICRO_INPUT_IDS)
static const std::vector<int32_t> kInputIds = {42, 105, 12, 88};

// ---------------------------------------------------------------------------
// Minimal NPY v1.0 loader (shared pattern from other Phase 4 tests)
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
// Structural / Construction Tests
// ---------------------------------------------------------------------------

TEST(ModelTest, ConstructFromWeightLoader) {
    bmgpt::WeightLoader loader(kMicroWeights);
    EXPECT_NO_THROW(bmgpt::Model model(loader, micro_config()));
}

TEST(ModelTest, ConfigIsStored) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Model        model(loader, micro_config());

    EXPECT_EQ(model.n_layers(),       2u);
    EXPECT_EQ(model.n_heads(),        2u);
    EXPECT_EQ(model.d_model(),       64u);
    EXPECT_EQ(model.vocab_size(),   256u);
    EXPECT_EQ(model.context_length(), 64u);
}

TEST(ModelTest, BlockCountMatchesNLayers) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Model        model(loader, micro_config());

    // Verify we can access block 0 and block 1 without throwing
    EXPECT_NO_THROW(model.block(0));
    EXPECT_NO_THROW(model.block(1));
    // Accessing block 2 should throw (out of range)
    EXPECT_THROW(model.block(2), std::out_of_range);
}

TEST(ModelTest, WteTShape) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Model        model(loader, micro_config());

    // wte_T_ must be [d_model, vocab_size]
    EXPECT_EQ(model.wte_T().ndim(),   2u);
    EXPECT_EQ(model.wte_T().size(0), 64u);   // d_model
    EXPECT_EQ(model.wte_T().size(1), 256u);  // vocab_size
}

// ---------------------------------------------------------------------------
// Output Shape Tests
// ---------------------------------------------------------------------------

TEST(ModelTest, OutputShapeT4) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Model        model(loader, micro_config());

    bmgpt::Tensor logits = model.forward(kInputIds);  // T=4

    EXPECT_EQ(logits.ndim(),   2u);
    EXPECT_EQ(logits.size(0),  4u);    // T
    EXPECT_EQ(logits.size(1), 256u);   // vocab_size
}

TEST(ModelTest, OutputShapeT1) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Model        model(loader, micro_config());

    bmgpt::Tensor logits = model.forward({0});  // T=1

    EXPECT_EQ(logits.ndim(),   2u);
    EXPECT_EQ(logits.size(0),  1u);
    EXPECT_EQ(logits.size(1), 256u);
}

// ---------------------------------------------------------------------------
// Error Handling Tests (delegated to Embedding, but confirm they propagate)
// ---------------------------------------------------------------------------

TEST(ModelTest, EmptyInputThrows) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Model        model(loader, micro_config());
    EXPECT_THROW(model.forward({}), std::invalid_argument);
}

TEST(ModelTest, OutOfRangeTokenThrows) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Model        model(loader, micro_config());
    EXPECT_THROW(model.forward({256}), std::out_of_range);
}

TEST(ModelTest, SequenceTooLongThrows) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Model        model(loader, micro_config());
    std::vector<int32_t> too_long(65, 0);  // context_length=64
    EXPECT_THROW(model.forward(too_long), std::length_error);
}

// ---------------------------------------------------------------------------
// Numerical Validation — Final LayerNorm
// ---------------------------------------------------------------------------

TEST(ModelNumerical, FinalLnOut_NumericalMatch) {
    // We test the ln_f sub-layer independently:
    // Run the model up to (but not including) the logit projection, then
    // compare the ln_f output against final_ln_out.npy.
    // Strategy: apply the same pipeline manually using exposed sub-layers.
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Model        model(loader, micro_config());

    // Run embedding + blocks manually
    bmgpt::Tensor h = model.embedding().forward(kInputIds);
    for (size_t i = 0; i < model.n_layers(); ++i) {
        h = model.block(i).forward(h);
    }
    // Apply final LayerNorm
    bmgpt::Tensor ln_out = model.ln_f().forward(h);

    std::vector<float> ref = load_npy(kDumpsDir + "/final_ln_out.npy");
    compare_tensors(ln_out, ref, 1e-4f);
}

// ---------------------------------------------------------------------------
// Numerical Validation — Full Logits
// ---------------------------------------------------------------------------

TEST(ModelNumerical, Logits_NumericalMatch) {
    // Full end-to-end: forward() on [42, 105, 12, 88] must match logits.npy.
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Model        model(loader, micro_config());

    bmgpt::Tensor logits = model.forward(kInputIds);

    std::vector<float> ref = load_npy(kDumpsDir + "/logits.npy");
    compare_tensors(logits, ref, 1e-4f);
}

// ---------------------------------------------------------------------------
// Argmax Validation — Next-Token Prediction
// ---------------------------------------------------------------------------

TEST(ModelNumerical, Argmax_NextToken_MatchesPyTorch) {
    // The argmax of the last-token row of logits must match the reference.
    // Reference: computed by dump_reference.py using the same micro model.
    // We load logits.npy, take row T-1=3, find argmax, and compare to ours.
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Model        model(loader, micro_config());

    bmgpt::Tensor logits = model.forward(kInputIds);  // [4, 256]

    // Our argmax of the last token row
    const size_t T          = logits.size(0);    // 4
    const size_t vocab      = logits.size(1);    // 256
    const float* last_row   = logits.data() + (T - 1) * vocab;
    size_t our_argmax = static_cast<size_t>(
        std::max_element(last_row, last_row + vocab) - last_row);

    // Reference argmax from logits.npy
    std::vector<float> ref = load_npy(kDumpsDir + "/logits.npy");
    const float* ref_last  = ref.data() + (T - 1) * vocab;
    size_t ref_argmax = static_cast<size_t>(
        std::max_element(ref_last, ref_last + vocab) - ref_last);

    EXPECT_EQ(our_argmax, ref_argmax)
        << "Next-token prediction mismatch: C++ predicts token "
        << our_argmax << ", PyTorch predicts " << ref_argmax;
}

