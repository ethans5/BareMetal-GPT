# -*- coding: utf-8 -*-
import regex
import unicodedata

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

def is_contraction(text, idx):
    contractions = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d",
                    "’s", "’t", "’re", "’ve", "’m", "’ll", "’d"]
    for c in contractions:
        if text.startswith(c, idx):
            # Must not be immediately followed by another letter (contraction is a whole unit)
            after = idx + len(c)
            if after >= len(text) or get_cat(ord(text[after])) != "L":
                return len(c)
    return 0

def split_gpt2_manual(text):
    if not text:
        return []
    
    # We first decode text into codepoints with their indices
    cps = [ord(c) for c in text]
    N = len(cps)
    pieces = []
    i = 0
    
    while i < N:
        # 1. Contractions: 's, 't, 're, 've, 'm, 'll, 'd
        clen = is_contraction(text, i)
        if clen > 0:
            pieces.append(text[i:i+clen])
            i += clen
            continue
            
        cp = cps[i]
        cat = get_cat(cp)
        
        # 2. Whitespace: \s+(?!\S)|\s+
        if cat == "S":
            # Check how many whitespace characters we have
            j = i
            while j < N and get_cat(cps[j]) == "S":
                j += 1
            # If j < N (there is a non-whitespace after), \s+(?!\S) matches all except the last one IF the last one is followed by non-space
            # Wait: in Python regex: \s+(?!\S) matches consecutive spaces up to the one before a non-space, BUT wait:
            # Look at '   Multiple': pieces are ['  ', ' Multiple']
            # That's because ' ?\p{L}+' matched ' Multiple' (with the leading space)!
            # So the spaces before that were '  '!
            # Let's verify how ' ?\p{L}+' interacts with leading spaces:
            # If cp is ' ' and i+1 < N and get_cat(cps[i+1]) in ("L", "N", "O") and cps[i+1] != ord(' '):
            # then the leading space belongs to the next token!
            pass
            
        # Let's verify with the exact regex behavior
        break
