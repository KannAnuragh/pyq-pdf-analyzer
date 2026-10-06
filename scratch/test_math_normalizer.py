import re
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'pdf_qa_agent'))
from ingest.math_normalizer import normalize_math_in_text

test_str = "Here is a text with an image <img src='test.png'> and some math like this."
out = normalize_math_in_text(test_str)
print("OUT 1:", out)

test_str2 = "A pure formula line with image:\n<img src='test.png'>"
out2 = normalize_math_in_text(test_str2)
print("OUT 2:", out2)

test_str3 = "Here is a text with \uE000HTMLTAG0\uE001 which shouldn't happen but let's test"
out3 = normalize_math_in_text(test_str3)
print("OUT 3:", out3)
