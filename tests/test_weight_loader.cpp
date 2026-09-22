/// @file test_weight_loader.cpp
/// @brief Unit tests for bmgpt::WeightLoader (Phase 2, Deliverable 3).
///
/// All tests target the micro model (micro_gpt2_weights.safetensors, ~473 KB).
/// The path to the weights directory is injected by CMake as WEIGHTS_DIR.
///
/// Micro model specs (from export_weights.py / dump_reference.py):
///   n_layers=2, n_heads=2, d_model=64, vocab_size=256, n_positions=64
///   n_inner=256 (= 4 * d_model)
///
/// Expected tensor shapes (post-Conv1D transposition applied at export time):
///   wte.weight                : [256, 64]
///   wpe.weight                : [64,  64]
///   ln_f.weight / ln_f.bias   : [64]
///   h.0.attn.c_attn.weight    : [192, 64]   (3*d_model, d_model)
///   h.0.attn.c_attn.bias      : [192]
///   h.0.attn.c_proj.weight    : [64,  64]
///   h.0.mlp.c_fc.weight       : [256, 64]
///   h.0.mlp.c_proj.weight     : [64, 256]

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

// Macro to stringify the define value
#define XSTR(x) STR(x)
#define STR(x) #x

static const std::string kWeightsDir  = XSTR(WEIGHTS_DIR);
static const std::string kMicroFile   = kWeightsDir + "/micro_gpt2_weights.safetensors";

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

class WeightLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        loader_ = std::make_unique<bmgpt::WeightLoader>(kMicroFile);
    }
    std::unique_ptr<bmgpt::WeightLoader> loader_;
};

// ---------------------------------------------------------------------------
// 1. Header parsing — tensor count
// ---------------------------------------------------------------------------

TEST_F(WeightLoaderTest, NumTensors) {
    // micro model has exactly 28 tensors (2 layers * 12 params + 4 global)
    EXPECT_EQ(loader_->num_tensors(), 28u);
}

// ---------------------------------------------------------------------------
// 2. Key existence
// ---------------------------------------------------------------------------

TEST_F(WeightLoaderTest, HasKey_ExistingKey) {
    EXPECT_TRUE(loader_->has("wte.weight"));
}

TEST_F(WeightLoaderTest, HasKey_NonExistingKey) {
    EXPECT_FALSE(loader_->has("this.does.not.exist"));
}

// ---------------------------------------------------------------------------
// 3. Full key list
// ---------------------------------------------------------------------------

TEST_F(WeightLoaderTest, KeysList_ContainsAllExpectedKeys) {
    // All 28 expected tensor names for a 2-layer micro model
    static const std::vector<std::string> kExpected = {
        "h.0.attn.c_attn.bias",   "h.0.attn.c_attn.weight",
        "h.0.attn.c_proj.bias",   "h.0.attn.c_proj.weight",
        "h.0.ln_1.bias",          "h.0.ln_1.weight",
        "h.0.ln_2.bias",          "h.0.ln_2.weight",
        "h.0.mlp.c_fc.bias",      "h.0.mlp.c_fc.weight",
        "h.0.mlp.c_proj.bias",    "h.0.mlp.c_proj.weight",
        "h.1.attn.c_attn.bias",   "h.1.attn.c_attn.weight",
        "h.1.attn.c_proj.bias",   "h.1.attn.c_proj.weight",
        "h.1.ln_1.bias",          "h.1.ln_1.weight",
        "h.1.ln_2.bias",          "h.1.ln_2.weight",
        "h.1.mlp.c_fc.bias",      "h.1.mlp.c_fc.weight",
        "h.1.mlp.c_proj.bias",    "h.1.mlp.c_proj.weight",
        "ln_f.bias",              "ln_f.weight",
        "wpe.weight",             "wte.weight",
    };

    auto actual = loader_->keys();  // returned sorted
    ASSERT_EQ(actual.size(), kExpected.size());
    for (const auto& key : kExpected) {
        EXPECT_TRUE(loader_->has(key)) << "Missing key: " << key;
    }
}

// ---------------------------------------------------------------------------
// 4. Shape verification
// ---------------------------------------------------------------------------

TEST_F(WeightLoaderTest, WteWeightShape) {
    auto t = loader_->load("wte.weight");
    ASSERT_EQ(t.ndim(), 2u);
    EXPECT_EQ(t.size(0), 256u);  // vocab_size
    EXPECT_EQ(t.size(1), 64u);   // d_model
}

TEST_F(WeightLoaderTest, WpeWeightShape) {
    auto t = loader_->load("wpe.weight");
    ASSERT_EQ(t.ndim(), 2u);
    EXPECT_EQ(t.size(0), 64u);   // n_positions
    EXPECT_EQ(t.size(1), 64u);   // d_model
}

TEST_F(WeightLoaderTest, LnFWeightShape) {
    auto t = loader_->load("ln_f.weight");
    ASSERT_EQ(t.ndim(), 1u);
    EXPECT_EQ(t.size(0), 64u);   // d_model
}

TEST_F(WeightLoaderTest, AttnCAttnWeightShape) {
    // After Conv1D transposition: [3*d_model, d_model] = [192, 64]
    auto t = loader_->load("h.0.attn.c_attn.weight");
    ASSERT_EQ(t.ndim(), 2u);
    EXPECT_EQ(t.size(0), 192u);  // 3 * d_model
    EXPECT_EQ(t.size(1), 64u);   // d_model
}

TEST_F(WeightLoaderTest, MlpCFcWeightShape) {
    // [4*d_model, d_model] = [256, 64]
    auto t = loader_->load("h.0.mlp.c_fc.weight");
    ASSERT_EQ(t.ndim(), 2u);
    EXPECT_EQ(t.size(0), 256u);  // d_ff = 4 * d_model
    EXPECT_EQ(t.size(1), 64u);   // d_model
}

TEST_F(WeightLoaderTest, MlpCProjWeightShape) {
    // [d_model, 4*d_model] = [64, 256]
    auto t = loader_->load("h.0.mlp.c_proj.weight");
    ASSERT_EQ(t.ndim(), 2u);
    EXPECT_EQ(t.size(0), 64u);   // d_model
    EXPECT_EQ(t.size(1), 256u);  // d_ff
}

// ---------------------------------------------------------------------------
// 5. Numeric values — wte.weight first 4 elements
//    Ground truth: obtained from the micro .safetensors file via Python.
//    wte.weight first 4 floats (seed=42):
//      [-0.008161831647157669, -0.03086434304714203,
//       0.0006699332734569907,  0.00378024997189641]
// ---------------------------------------------------------------------------

TEST_F(WeightLoaderTest, WteWeightValues_First4) {
    auto t = loader_->load("wte.weight");

    ASSERT_GE(t.numel(), 4u);

    const float expected[4] = {
        -0.008161831647157669f,
        -0.030864343047142029f,
         0.000669933273456990f,
         0.003780249971896410f,
    };

    for (int i = 0; i < 4; ++i) {
        float actual = t.data()[i];
        EXPECT_NEAR(actual, expected[i], 1e-6f)
            << "Mismatch at wte.weight[" << i << "]: "
            << actual << " vs " << expected[i];
    }
}

// ---------------------------------------------------------------------------
// 6. Total element count
// ---------------------------------------------------------------------------

TEST_F(WeightLoaderTest, WteWeightNumel) {
    auto t = loader_->load("wte.weight");
    EXPECT_EQ(t.numel(), 256u * 64u);
}

// ---------------------------------------------------------------------------
// 7. Error handling — file not found
// ---------------------------------------------------------------------------

TEST(WeightLoaderErrorTest, FileNotFound) {
    EXPECT_THROW(
        bmgpt::WeightLoader loader("/nonexistent/path/weights.safetensors"),
        std::runtime_error
    );
}

// ---------------------------------------------------------------------------
// 8. Error handling — load unknown key
// ---------------------------------------------------------------------------

TEST(WeightLoaderErrorTest, LoadUnknownKey) {
    bmgpt::WeightLoader loader(kMicroFile);
    EXPECT_THROW(
        loader.load("nonexistent.tensor"),
        std::runtime_error
    );
}
