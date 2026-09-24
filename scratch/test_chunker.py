import re

raw_text = """1. This is a question
(A) option A
(B) option B"""

q_extract_pattern = r'^[\*\_]*(\d+)[\*\_]*\s*(?:[\.\)]|:-)\s*[\*\_]*\s*(.*)'

q_match = re.match(q_extract_pattern, raw_text, flags=re.DOTALL | re.IGNORECASE)
if q_match:
    print(f"Num: {q_match.group(1)}")
    print(f"Stem: {q_match.group(2)[:20]}")
else:
    print("NO MATCH")

