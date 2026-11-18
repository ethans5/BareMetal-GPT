# -*- coding: utf-8 -*-
"""
download_tokenizer.py — Download official GPT-2 vocab.json and merges.txt
and save them to weights/ directory.
"""
import os
import sys
import urllib.request

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

FILES = {
    "vocab.json": "https://huggingface.co/gpt2/raw/main/vocab.json",
    "merges.txt": "https://huggingface.co/gpt2/raw/main/merges.txt"
}

def main():
    for filename, url in FILES.items():
        dest = os.path.join(WEIGHTS_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[SKIP] {filename} already exists ({os.path.getsize(dest)} bytes)")
            continue
        print(f"[DOWNLOADING] {filename} from {url} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"[OK] {filename} saved ({os.path.getsize(dest)} bytes)")

if __name__ == "__main__":
    main()
