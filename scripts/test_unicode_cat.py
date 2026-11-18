# -*- coding: utf-8 -*-
import unicodedata

def unicode_category(cp):
    ch = chr(cp)
    cat = unicodedata.category(ch)
    if cat.startswith("L"):
        return "L"
    if cat.startswith("N"):
        return "N"
    if cat.startswith("Z") or ch in "\t\n\r\f\v":
        return "S"
    return "O"

# Let's verify on test chars:
test_cps = [
    ord('a'), ord('Z'), ord('9'), ord(' '), ord('\n'),
    ord('é'), ord('ç'), ord('ü'), ord('ñ'), ord('€'),
    0x4E2D, # 中
    0x3053, # こ
    0x041F, # П
    0x1F680 # 🚀
]

for cp in test_cps:
    print(f"U+{cp:04X} ({chr(cp)}): {unicode_category(cp)} (raw: {unicodedata.category(chr(cp))})")
