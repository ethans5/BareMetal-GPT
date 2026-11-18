# -*- coding: utf-8 -*-
import regex

GPT2_SPLIT_REGEX = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
pat = regex.compile(GPT2_SPLIT_REGEX)

test_strings = [
    "Hello world!",
    "It's a test, isn't it? 123.",
    "   Multiple    spaces\nand\tnewlines!",
    "Bonjour, ça va? L'été à Paris coût 100€.",
    "def add(a: int, b: int) -> int:\n    return a + b\n"
]

for s in test_strings:
    pieces = pat.findall(s)
    print(f"Original: {repr(s)}")
    print(f"Pieces: {pieces}")
    print(f"Reconstructed == Original: {''.join(pieces) == s}")
    print("-" * 50)
