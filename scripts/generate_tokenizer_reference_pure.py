# -*- coding: utf-8 -*-
"""
Pure Python implementation of GPT-2 Tokenizer (no transformers dependency).
Generates test reference for C++ unit tests.
"""
import json
import os
import regex

def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

class SimpleGPT2Tokenizer:
    def __init__(self, vocab_file, merges_file):
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        with open(merges_file, "r", encoding="utf-8") as f:
            bpe_data = f.read().split("\n")
        
        # merges.txt starts with #version: 0.2
        if bpe_data[0].startswith("#"):
            bpe_data = bpe_data[1:]
        
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data if merge_str.strip()]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        
        self.pat = regex.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        self.cache = {}

    def get_pairs(self, word):
        """Return set of symbol pairs in a word (tuple of strings)."""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = self.get_pairs(word)
        if not pairs:
            return token

        while True:
            # Find the pair with the lowest rank in bpe_ranks
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        for token in regex.findall(self.pat, text):
            # Encode each byte of utf-8 representation to mapped unicode char
            token_bytes = token.encode("utf-8")
            token_translated = "".join(self.byte_encoder[b] for b in token_bytes)
            # Apply BPE
            bpe_result = self.bpe(token_translated).split(" ")
            for bpe_token in bpe_result:
                bpe_tokens.append(self.encoder[bpe_token])
        return bpe_tokens

    def decode(self, tokens):
        text = "".join([self.decoder[token] for token in tokens])
        byte_arr = bytearray([self.byte_decoder[c] for c in text])
        return byte_arr.decode("utf-8", errors="replace")

def main():
    tok = SimpleGPT2Tokenizer("weights/vocab.json", "weights/merges.txt")
    
    test_cases = [
        "Hello world!",
        "It's a test, isn't it? 123.",
        "   Multiple    spaces\nand\tnewlines!",
        "Bonjour, ça va? L'été à Paris coûte 100€.",
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        "A",
        " ",
        "  ",
        "\n",
        "\n\n",
        "The quick brown fox jumps over the lazy dog.",
        "BareMetal-GPT is a from-scratch C++17 inference engine.",
        "1234567890 +-/*=(){}[]<>:;.,?!@#$%^&*",
        "Unicode: 🚀 café naïve résumé 日本語 Русский",
        ""
    ]
    
    results = []
    for text in test_cases:
        tokens = tok.encode(text) if text else []
        decoded = tok.decode(tokens) if tokens else ""
        assert decoded == text, f"Mismatch for {repr(text)} != {repr(decoded)}"
        results.append({
            "text": text,
            "tokens": tokens,
            "decoded": decoded
        })
        print(f"Text: {repr(text)}")
        print(f"Tokens ({len(tokens)}): {tokens}")
        print(f"Roundtrip OK: {decoded == text}")
        print("-" * 50)
        
    os.makedirs("reference_dumps", exist_ok=True)
    with open("reference_dumps/tokenizer_reference.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("All tests passed! Reference saved to reference_dumps/tokenizer_reference.json")

if __name__ == "__main__":
    main()
