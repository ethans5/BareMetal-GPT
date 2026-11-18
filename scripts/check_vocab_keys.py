# -*- coding: utf-8 -*-
import json

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

b2u = bytes_to_unicode()
u2b = {v: k for k, v in b2u.items()}

with open("weights/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Test some tokens
print(f"'ĠThe' in vocab: {'ĠThe' in vocab}, id: {vocab.get('ĠThe')}")
print(f"'The' in vocab: {'The' in vocab}, id: {vocab.get('The')}")
print(f"'Ġworld' in vocab: {'Ġworld' in vocab}, id: {vocab.get('Ġworld')}")
print(f"'!' in vocab: {'!' in vocab}, id: {vocab.get('!')}")
