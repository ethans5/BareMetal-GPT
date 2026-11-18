# -*- coding: utf-8 -*-
import regex
import json

pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

with open("reference_dumps/tokenizer_reference.json", "r", encoding="utf-8") as f:
    cases = json.load(f)

for case in cases:
    text = case["text"]
    pieces = pat.findall(text)
    print(f"TEXT: {repr(text)}")
    print(f"PIECES: {pieces}")
    print()
