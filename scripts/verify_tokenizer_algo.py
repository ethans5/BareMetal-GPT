# -*- coding: utf-8 -*-
import regex
import unicodedata

pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def get_cat(cp):
    ch = chr(cp)
    cat = unicodedata.category(ch)
    if cat.startswith("L"):
        return "L"
    if cat.startswith("N"):
        return "N"
    if cat.startswith("Z") or ch in "\t\n\r\f\v":
        return "S"
    return "O"

def match_next_token(text, start_idx):
    N = len(text)
    if start_idx >= N:
        return ""
    
    # Branch 1: Contractions 's|'t|'re|'ve|'m|'ll|'d
    contractions = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
    for c in contractions:
        if text.startswith(c, start_idx):
            return c
            
    # Check leading space
    has_space = (text[start_idx] == ' ')
    idx = start_idx + (1 if has_space else 0)
    
    if idx < N:
        cp = ord(text[idx])
        cat = get_cat(cp)
        
        # Branch 2: ?\p{L}+
        if cat == "L":
            end = idx + 1
            while end < N and get_cat(ord(text[end])) == "L":
                end += 1
            return text[start_idx:end]
            
        # Branch 3: ?\p{N}+
        if cat == "N":
            end = idx + 1
            while end < N and get_cat(ord(text[end])) == "N":
                end += 1
            return text[start_idx:end]
            
        # Branch 4: ?[^\s\p{L}\p{N}]+
        if cat == "O":
            end = idx + 1
            while end < N and get_cat(ord(text[end])) == "O":
                end += 1
            return text[start_idx:end]

    # Branch 5 & 6: \s+(?!\S)|\s+
    # Matches consecutive whitespace characters
    end = start_idx
    while end < N and get_cat(ord(text[end])) == "S":
        end += 1
    
    # If this whitespace run ends right before a non-whitespace (\S),
    # then if the last whitespace in the run is ' ', that last ' ' can be claimed
    # by the following token via ' ?\p{L}+', etc.!
    # So \s+(?!\S) matches up to end - 1 if text[end-1] == ' ' and end < N.
    if end > start_idx:
        if end < N and get_cat(ord(text[end])) != "S" and text[end - 1] == ' ':
            # Leave the single space for the next token, unless the run is only 1 space and followed by something
            # Wait: if text is ' Hello', start_idx=0, has_space=True, cat='L', handled by Branch 2 above!
            # If we reached here, has_space was True, but next char was ALSO a space!
            # So the last space in the run of spaces is left for the next token!
            if end - 1 > start_idx:
                return text[start_idx:end - 1]
        return text[start_idx:end]

    # Fallback (single character)
    return text[start_idx:start_idx+1]

def tokenize_manual(text):
    res = []
    i = 0
    while i < len(text):
        tok = match_next_token(text, i)
        if not tok:
            break
        res.append(tok)
        i += len(tok)
    return res

test_cases = [
    "Hello world!",
    "It's a test, isn't it? 123.",
    "   Multiple    spaces\nand\tnewlines!",
    "Bonjour, ça va? L'été à Paris coûte 100€.",
    "def add(a: int, b: int) -> int:\n    return a + b\n",
    "A", " ", "  ", "   ", "\n", "\n\n",
    "The quick brown fox jumps over the lazy dog.",
    "BareMetal-GPT is a from-scratch C++17 inference engine.",
    "1234567890 +-/*=(){}[]<>:;.,?!@#$%^&*",
    "Unicode: 🚀 café naïve résumé 日本語 Русский"
]

all_match = True
for t in test_cases:
    expected = pat.findall(t)
    actual = tokenize_manual(t)
    if expected != actual:
        all_match = False
        print(f"FAILED for: {repr(t)}")
        print(f"Expected: {expected}")
        print(f"Actual:   {actual}")
        break

if all_match:
    print("SUCCESS! 100% exact match on all test cases!")
