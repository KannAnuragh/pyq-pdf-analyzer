import re

pat1 = r"^\s*(?:Answer|Ans|Solution|Sol|Correct\s*Option|Option|Key)[\s\:\-\.]*[\(\[]?\s*([A-D])\s*[\)\]]?(?:[\s\.\:\-]+|is\s*correct\b)?"
pat2 = r"^\s*(?:[\(\[]\s*([A-D])\s*[\)\]]|([A-D])\s*[\.\:\-\)]+)(?=\s|$)"

test_cases = [
    "C) E != 0",
    "A - The value is",
    "B: This is because",
]

for t in test_cases:
    m1 = re.match(pat1, t, re.IGNORECASE)
    m2 = re.match(pat2, t, re.IGNORECASE)
    if m1:
        print(f"'{t}' -> pat1 matches: {m1.group(1)}")
    elif m2:
        print(f"'{t}' -> pat2 matches: {m2.group(1) or m2.group(2)}")
    else:
        print(f"'{t}' -> NO MATCH")
