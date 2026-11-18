# -*- coding: utf-8 -*-
import json

with open("weights/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

with open("weights/merges.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    merges = [line.strip() for line in lines if line.strip() and not line.startswith("#")]

print(f"vocab_size: {len(vocab)}")
print(f"merges_count: {len(merges)}")
print(f"first 3 merges: {merges[:3]}")
print(f"eos token: <|endoftext|> -> {vocab.get('<|endoftext|>')}")
