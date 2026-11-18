/// @file test_tokenizer.cpp
/// @brief Unit tests and HuggingFace reference validation for bmgpt::Tokenizer (Phase 5).

#include "baremetalgpt/Tokenizer.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#ifndef WEIGHTS_DIR
#  error "WEIGHTS_DIR must be defined via target_compile_definitions"
#endif
#ifndef DUMPS_DIR
#  error "DUMPS_DIR must be defined via target_compile_definitions"
#endif

#define XSTR(x) STR(x)
#define STR(x)  #x

static const std::string kWeightsDir = XSTR(WEIGHTS_DIR);
static const std::string kDumpsDir   = XSTR(DUMPS_DIR);
static const std::string kVocabPath  = kWeightsDir + "/vocab.json";
static const std::string kMergesPath = kWeightsDir + "/merges.txt";

// ── Basic Structural Tests ───────────────────────────────────────────────────

TEST(TokenizerTest, LoadVocabAndMerges) {
    EXPECT_NO_THROW({
        bmgpt::Tokenizer tok(kVocabPath, kMergesPath);
        EXPECT_EQ(tok.vocab_size(), 50257u);
        EXPECT_EQ(tok.num_merges(), 49992u);
        EXPECT_EQ(tok.eos_token_id(), 50256);
    });
}

TEST(TokenizerTest, MissingFileThrows) {
    EXPECT_THROW(bmgpt::Tokenizer("non_existent_vocab.json", kMergesPath), std::runtime_error);
    EXPECT_THROW(bmgpt::Tokenizer(kVocabPath, "non_existent_merges.txt"), std::runtime_error);
}

TEST(TokenizerTest, ByteToUnicodeBijection) {
    bmgpt::Tokenizer tok(kVocabPath, kMergesPath);

    // Every byte 0..255 must map to a non-empty string and map back to itself
    std::unordered_set<std::string> seen_unicode;
    for (int b = 0; b < 256; ++b) {
        uint8_t byte_val = static_cast<uint8_t>(b);
        const std::string& ustr = tok.byte_to_unicode(byte_val);
        EXPECT_FALSE(ustr.empty());

        // Check uniqueness of mapping
        EXPECT_TRUE(seen_unicode.find(ustr) == seen_unicode.end());
        seen_unicode.insert(ustr);

        // Check reverse lookup
        int16_t rev_byte = tok.unicode_to_byte(ustr);
        EXPECT_EQ(rev_byte, byte_val);
    }
    EXPECT_EQ(seen_unicode.size(), 256u);
}

// ── Common Vocabulary Entries & Pieces ───────────────────────────────────────

TEST(TokenizerTest, KnownTokens) {
    bmgpt::Tokenizer tok(kVocabPath, kMergesPath);

    EXPECT_EQ(tok.piece_to_token("!"), 0);
    EXPECT_EQ(tok.piece_to_token("ĠThe"), 383);
    EXPECT_EQ(tok.piece_to_token("The"), 464);
    EXPECT_EQ(tok.piece_to_token("Ġworld"), 995);
    EXPECT_EQ(tok.piece_to_token("<|endoftext|>"), 50256);

    EXPECT_EQ(tok.token_to_piece(0), "!");
    EXPECT_EQ(tok.token_to_piece(383), "ĠThe");
    EXPECT_EQ(tok.token_to_piece(464), "The");
    EXPECT_EQ(tok.token_to_piece(995), "Ġworld");
    EXPECT_EQ(tok.token_to_piece(50256), "<|endoftext|>");
}

// ── Error Handling Tests ─────────────────────────────────────────────────────

TEST(TokenizerTest, InvalidTokenIdThrowsOnDecode) {
    bmgpt::Tokenizer tok(kVocabPath, kMergesPath);

    EXPECT_THROW(tok.decode({-1}), std::out_of_range);
    EXPECT_THROW(tok.decode({50257}), std::out_of_range);
    EXPECT_THROW(tok.decode({100000}), std::out_of_range);
}

TEST(TokenizerTest, InvalidPieceThrows) {
    bmgpt::Tokenizer tok(kVocabPath, kMergesPath);

    EXPECT_THROW(tok.piece_to_token("non_existent_token_piece_xyz_12345"), std::out_of_range);
}

// ── Lossless Roundtrip Invariant ─────────────────────────────────────────────

TEST(TokenizerTest, RoundtripTexts) {
    bmgpt::Tokenizer tok(kVocabPath, kMergesPath);

    const std::vector<std::string> test_texts = {
        "",
        "A",
        "Hello",
        "Hello world!",
        "It's a test, isn't it? 123.",
        "   Multiple    spaces\nand\tnewlines!",
        "Bonjour, ça va? L'été à Paris coûte 100€.",
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        "The quick brown fox jumps over the lazy dog.",
        "BareMetal-GPT is a from-scratch C++17 inference engine.",
        "1234567890 +-/*=(){}[]<>:;.,?!@#$%^&*",
        "Unicode: 🚀 café naïve résumé 日本語 Русский",
        "Special: <|endoftext|>"
    };

    for (const auto& text : test_texts) {
        std::vector<int32_t> tokens = tok.encode(text);
        std::string decoded = tok.decode(tokens);
        EXPECT_EQ(decoded, text) << "Roundtrip failed for: " << text;
    }
}

// ── Numerical Match with Reference Dumps ─────────────────────────────────────

TEST(TokenizerNumerical, MatchReferenceDump) {
    bmgpt::Tokenizer tok(kVocabPath, kMergesPath);

    // Load reference_dumps/tokenizer_reference.json
    std::string ref_path = kDumpsDir + "/../tokenizer_reference.json";
    std::ifstream file(ref_path);
    ASSERT_TRUE(file.is_open()) << "Cannot open " << ref_path;

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    // Verify key test cases explicitly
    // 1. "Hello world!" -> [15496, 995, 0]
    {
        std::vector<int32_t> actual = tok.encode("Hello world!");
        std::vector<int32_t> expected = {15496, 995, 0};
        EXPECT_EQ(actual, expected);
        EXPECT_EQ(tok.decode(actual), "Hello world!");
    }

    // 2. "The quick brown fox jumps over the lazy dog."
    {
        std::vector<int32_t> actual = tok.encode("The quick brown fox jumps over the lazy dog.");
        std::vector<int32_t> expected = {464, 2068, 7586, 21831, 18045, 625, 262, 16931, 3290, 13};
        EXPECT_EQ(actual, expected);
        EXPECT_EQ(tok.decode(actual), "The quick brown fox jumps over the lazy dog.");
    }

    // 3. "It's a test, isn't it? 123."
    {
        std::vector<int32_t> actual = tok.encode("It's a test, isn't it? 123.");
        std::vector<int32_t> expected = {1026, 338, 257, 1332, 11, 2125, 470, 340, 30, 17031, 13};
        EXPECT_EQ(actual, expected);
        EXPECT_EQ(tok.decode(actual), "It's a test, isn't it? 123.");
    }

    // 4. "   Multiple    spaces\nand\tnewlines!"
    {
        std::vector<int32_t> actual = tok.encode("   Multiple    spaces\nand\tnewlines!");
        std::vector<int32_t> expected = {220, 220, 20401, 220, 220, 220, 9029, 198, 392, 197, 3605, 6615, 0};
        EXPECT_EQ(actual, expected);
        EXPECT_EQ(tok.decode(actual), "   Multiple    spaces\nand\tnewlines!");
    }

    // 5. Python code
    {
        std::string code = "def add(a: int, b: int) -> int:\n    return a + b\n";
        std::vector<int32_t> actual = tok.encode(code);
        std::vector<int32_t> expected = {4299, 751, 7, 64, 25, 493, 11, 275, 25, 493, 8, 4613, 493, 25, 198, 220, 220, 220, 1441, 257, 1343, 275, 198};
        EXPECT_EQ(actual, expected);
        EXPECT_EQ(tok.decode(actual), code);
    }
}
