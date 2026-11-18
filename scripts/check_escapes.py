# -*- coding: utf-8 -*-
import json

with open("weights/vocab.json", "r", encoding="utf-8") as f:
    text = f.read()

escapes = set()
i = 0
while i < len(text):
    if text[i] == '\\':
        escapes.add(text[i:i+2])
        i += 2
    else:
        i += 1

print(f"All escape sequences in vocab.json: {escapes}")
