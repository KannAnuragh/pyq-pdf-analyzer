import re

text = """1. An electric dipole of moment p is placed parallel to the uniform electric field. The
amount of work done in rotating the dipole by 90°:
[1 mark]
(a) 2pE
(b) pE
(c) pE/2
(d) Zero
Answer:
(b) pE

2. Three charges 2q, q and q lie at vertices of a triangle."""

q_prefix_pattern = r'[\*\_]*\d+[\*\_]*\s*(?:[\.\)]|:-)\s*[\*\_]*\s*'
q_extract_pattern = r'^[\*\_]*(\d+)[\*\_]*\s*(?:[\.\)]|:-)\s*[\*\_]*\s*(.*)'

blocks = re.split(fr'(?:\n|\s+|^)(?={q_prefix_pattern})', text, flags=re.MULTILINE | re.IGNORECASE)

valid = 0
for b in blocks:
    b = b.strip()
    if not b: continue
    print(f"Checking block: {b[:20]}...")
    q_match = re.match(q_extract_pattern, b, flags=re.DOTALL | re.IGNORECASE)
    if q_match:
        print(f"  Match! Num: {q_match.group(1)}")
        valid += 1
    else:
        print("  No match!")
print(f"Valid questions: {valid}")
