# -*- coding: utf-8 -*-
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

b2u = bytes_to_unicode()
print(f"Total mappings: {len(b2u)}")
# Let's inspect some mappings
print(f"byte 32 (' '): {ord(b2u[32])} -> {b2u[32]} ({repr(b2u[32])})")
print(f"byte 10 ('\\n'): {ord(b2u[10])} -> {b2u[10]} ({repr(b2u[10])})")

# Let's check the exact formula
bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
cs = bs[:]
n = 0
for b in range(2**8):
    if b not in bs:
        bs.append(b)
        cs.append(2**8 + n)
        n += 1

matches = True
for b, c in zip(bs, cs):
    if chr(c) != b2u[b]:
        matches = False
        print(f"Mismatch for byte {b}: expected {chr(c)} got {b2u[b]}")
        break

print(f"Formula exact match: {matches}")
