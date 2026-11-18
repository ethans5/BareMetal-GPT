# -*- coding: utf-8 -*-
import json
import os
from transformers import GPT2Tokenizer

vocab_path = os.path.abspath("weights/vocab.json")
merges_path = os.path.abspath("weights/merges.txt")

tokenizer = GPT2Tokenizer(vocab_file=vocab_path, merges_file=merges_path)

test_cases = [
    "Hello world!",
    "It's a test, isn't it? 123.",
    "   Multiple    spaces\nand\tnewlines!",
    "Bonjour, ça va? L'été à Paris coûte 100€.",
    "def add(a: int, b: int) -> int:\n    return a + b\n",
    "Special tokens: <|endoftext|>",
    "A",
    " ",
    "  ",
    "\n",
    "\n\n",
    "The quick brown fox jumps over the lazy dog.",
    "BareMetal-GPT is a from-scratch C++17 inference engine.",
    "1234567890 +-/*=(){}[]<>:;.,?!@#$%^&*",
    "Unicode: 🚀 café naïve résumé 日本語 Русский",
    ""  # Empty string
]

results = []
for text in test_cases:
    tokens = tokenizer.encode(text, add_special_tokens=False) if text else []
    decoded = tokenizer.decode(tokens) if tokens else ""
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

print("Saved reference_dumps/tokenizer_reference.json successfully!")
