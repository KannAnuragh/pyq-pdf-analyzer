import re
pat1 = r"^\s*(?:Answer|Ans|Solution|Sol|Correct\s*Option|Option|Key)[\s\:\-\.]*[\(\[]?\s*([A-D])\s*[\)\]]?(?:[\s\.\:\-]+|is\s*correct\b)?"
t = "Answer: Because it is gravity"
m = re.match(pat1, t, re.IGNORECASE)
if m:
    print(f"Matched: '{m.group(0)}', extracted letter: '{m.group(1)}', remainder: '{t[m.end():]}'")
else:
    print("NO MATCH")
