/// @file test_embedding.cpp
/// @brief Numerical validation and unit tests for bmgpt::Embedding (Phase 4).
///
/// Strategy:
///   - Load micro-model weights via WeightLoader (vocab_size=256, d_model=64,
///     context_length=64).
///   - Construct an Embedding instance and verify structural properties.
///   - Verify all error-handling paths (empty ids, out-of-range, too-long sequence).
///   - Run forward() on the canonical test sequence [42, 105, 12, 88] and compare
///     the output against three reference dumps generated in Phase 2:
///       token_embeddings.npy     == wte lookup (before wpe addition)
///       position_embeddings.npy  == wpe lookup (before wte addition)
///       initial_hidden.npy       == wte + wpe   (forward() output)
///   - Tolerance: 1e-6 (embeddings are simple lookups — no floating-point
///     accumulation — so the error is essentially rounding only).

#include "baremetalgpt/Embedding.hpp"
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
#define STR(x) #x

static const std::string kWeightsDir   = XSTR(WEIGHTS_DIR);
static const std::string kDumpsDir     = XSTR(DUMPS_DIR);
static const std::string kMicroWeights = kWeightsDir + "/micro_gpt2_weights.safetensors";

// ---------------------------------------------------------------------------
// Minimal NPY v1.0 loader (shared pattern from Phase 3 tests)
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

static void compare_tensors(const bmgpt::Tensor& actual,
                            const std::vector<float>& expected,
                            float tol = 1e-6f)
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
// Canonical test sequence (must match dump_reference.py MICRO_INPUT_IDS)
// ---------------------------------------------------------------------------

static const std::vector<int32_t> kInputIds = {42, 105, 12, 88};

// ---------------------------------------------------------------------------
// Structural & Dimension Tests
// ---------------------------------------------------------------------------

TEST(EmbeddingTest, ConstructFromWeightLoader) {
    bmgpt::WeightLoader loader(kMicroWeights);
    EXPECT_NO_THROW(bmgpt::Embedding emb(loader));
}

TEST(EmbeddingTest, DimensionsFromMicroWeights) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Embedding emb(loader);

    EXPECT_EQ(emb.vocab_size(),     256u);
    EXPECT_EQ(emb.context_length(),  64u);
    EXPECT_EQ(emb.d_model(),         64u);
}

TEST(EmbeddingTest, WteShapeIsCorrect) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Embedding emb(loader);

    EXPECT_EQ(emb.wte().ndim(),   2u);
    EXPECT_EQ(emb.wte().size(0), 256u);  // vocab_size
    EXPECT_EQ(emb.wte().size(1),  64u);  // d_model
}

TEST(EmbeddingTest, WpeShapeIsCorrect) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Embedding emb(loader);

    EXPECT_EQ(emb.wpe().ndim(),   2u);
    EXPECT_EQ(emb.wpe().size(0),  64u);  // context_length
    EXPECT_EQ(emb.wpe().size(1),  64u);  // d_model
}

TEST(EmbeddingTest, OutputShapeMatchesInputLength) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Embedding emb(loader);

    bmgpt::Tensor out4 = emb.forward(kInputIds);
    EXPECT_EQ(out4.ndim(),   2u);
    EXPECT_EQ(out4.size(0),  4u);   // T = 4 tokens
    EXPECT_EQ(out4.size(1), 64u);   // d_model

    bmgpt::Tensor out1 = emb.forward({0});
    EXPECT_EQ(out1.ndim(),   2u);
    EXPECT_EQ(out1.size(0),  1u);   // T = 1 token
    EXPECT_EQ(out1.size(1), 64u);
}

// ---------------------------------------------------------------------------
// Error Handling Tests
// ---------------------------------------------------------------------------

TEST(EmbeddingTest, EmptyInputIdsThrows) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Embedding emb(loader);

    EXPECT_THROW(emb.forward({}), std::invalid_argument);
}

TEST(EmbeddingTest, NegativeTokenIdThrows) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Embedding emb(loader);

    EXPECT_THROW(emb.forward({-1}),        std::out_of_range);
    EXPECT_THROW(emb.forward({0, -5, 10}), std::out_of_range);
}

TEST(EmbeddingTest, TokenIdAtVocabSizeThrows) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Embedding emb(loader);

    // vocab_size = 256, so ID 256 is one past the end
    EXPECT_THROW(emb.forward({256}), std::out_of_range);
}

TEST(EmbeddingTest, TokenIdWayOutOfRangeThrows) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Embedding emb(loader);

    EXPECT_THROW(emb.forward({50257}), std::out_of_range);
}

TEST(EmbeddingTest, SequenceTooLongThrows) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Embedding emb(loader);

    // context_length = 64 for micro model; 65 tokens must throw
    std::vector<int32_t> too_long(65, 0);
    EXPECT_THROW(emb.forward(too_long), std::length_error);
}

TEST(EmbeddingTest, MaxLengthSequenceDoesNotThrow) {
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Embedding emb(loader);

    // Exactly context_length = 64 tokens: must succeed
    std::vector<int32_t> max_len(64, 0);
    EXPECT_NO_THROW(emb.forward(max_len));
}

TEST(EmbeddingTest, DirectConstructorDimensionMismatchThrows) {
    // wte has d_model=64, wpe has d_model=32 — must throw
    bmgpt::Tensor wte({16, 64});
    bmgpt::Tensor wpe({8,  32});
    EXPECT_THROW(bmgpt::Embedding(wte, wpe), std::invalid_argument);
}

TEST(EmbeddingTest, DirectConstructorNon2DWteThrows) {
    bmgpt::Tensor wte({16, 8, 64});  // 3-D — invalid
    bmgpt::Tensor wpe({8, 64});
    EXPECT_THROW(bmgpt::Embedding(wte, wpe), std::invalid_argument);
}

TEST(EmbeddingTest, DirectConstructorNon2DWpeThrows) {
    bmgpt::Tensor wte({16, 64});
    bmgpt::Tensor wpe({64});         // 1-D — invalid
    EXPECT_THROW(bmgpt::Embedding(wte, wpe), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Numerical Validation: Token Embeddings (wte lookup)
// ---------------------------------------------------------------------------

TEST(EmbeddingNumerical, TokenEmbeddings_NumericalMatch) {
    // Verify that the first step (wte row-extraction) is bit-accurate.
    // We extract the wte rows manually and compare to token_embeddings.npy.
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Embedding emb(loader);

    std::vector<float> ref = load_npy(kDumpsDir + "/token_embeddings.npy");

    // Build a [4, 64] tensor by manually extracting wte rows
    const size_t D = emb.d_model();
    const float* wte_data = emb.wte().data();
    std::vector<float> manual_tok(kInputIds.size() * D);
    for (size_t i = 0; i < kInputIds.size(); ++i) {
        const size_t row = static_cast<size_t>(kInputIds[i]);
        for (size_t d = 0; d < D; ++d) {
            manual_tok[i * D + d] = wte_data[row * D + d];
        }
    }
    ASSERT_EQ(manual_tok.size(), ref.size());

    float max_err = 0.0f;
    for (size_t i = 0; i < manual_tok.size(); ++i) {
        float err = std::abs(manual_tok[i] - ref[i]);
        if (err > max_err) max_err = err;
        EXPECT_NEAR(manual_tok[i], ref[i], 1e-6f)
            << "Token embedding mismatch at flat index " << i;
    }
}

// ---------------------------------------------------------------------------
// Numerical Validation: Position Embeddings (wpe lookup)
// ---------------------------------------------------------------------------

TEST(EmbeddingNumerical, PositionEmbeddings_NumericalMatch) {
    // Verify that the wpe positional rows match position_embeddings.npy.
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Embedding emb(loader);

    std::vector<float> ref = load_npy(kDumpsDir + "/position_embeddings.npy");

    // Build a [4, 64] tensor by extracting wpe rows 0..3
    const size_t T = kInputIds.size();
    const size_t D = emb.d_model();
    const float* wpe_data = emb.wpe().data();
    std::vector<float> manual_pos(T * D);
    for (size_t i = 0; i < T; ++i) {
        for (size_t d = 0; d < D; ++d) {
            manual_pos[i * D + d] = wpe_data[i * D + d];
        }
    }
    ASSERT_EQ(manual_pos.size(), ref.size());

    for (size_t i = 0; i < manual_pos.size(); ++i) {
        EXPECT_NEAR(manual_pos[i], ref[i], 1e-6f)
            << "Position embedding mismatch at flat index " << i;
    }
}

// ---------------------------------------------------------------------------
// Numerical Validation: Initial Hidden State (forward output = wte + wpe)
// ---------------------------------------------------------------------------

TEST(EmbeddingNumerical, InitialHidden_NumericalMatch) {
    // This is the main integration test: forward() == initial_hidden.npy.
    bmgpt::WeightLoader loader(kMicroWeights);
    bmgpt::Embedding emb(loader);

    std::vector<float> ref = load_npy(kDumpsDir + "/initial_hidden.npy");

    bmgpt::Tensor out = emb.forward(kInputIds);

    // Shape check
    ASSERT_EQ(out.ndim(),   2u);
    ASSERT_EQ(out.size(0),  4u);
    ASSERT_EQ(out.size(1), 64u);

    // Numerical match — 1e-6 tolerance (lookup + addition, minimal accumulated error)
    compare_tensors(out, ref, 1e-6f);
}
