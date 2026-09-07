import re

for opt_text in ["Correct Answer:- Option-D", "Ans: B", "Answer: C", "Correct answer is D", "Answer A"]:
    ans_match = re.search(r'(?:Ans(?:wer)?|Correct(?:\s*Answer)?)\s*[:\-]*\s*(?:is\s*)?(?:Option\s*[:\-]*\s*)?([A-D])\b', opt_text, re.IGNORECASE)
    if ans_match:
        print(opt_text, "->", ans_match.group(1).upper())
