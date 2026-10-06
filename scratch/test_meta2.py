import re

def parse_trailing_meta(text: str):
    if not text:
        return "", None, None

    clean_text = text.strip()
    marked_answer = None
    explanation = None

    ans_pat = re.compile(
        r"(?:^|\n|\s+)\*?\*?(?:Answer|Ans(?:wer)?|Correct(?:\s*Answer)?|ഉത്തരം)[\*\:\-\s]*(?:is\s*)?(?:Option\s*[:\-]*\s*)?(?:\()?([A-D])(?:\))?(?=\s|$)",
        re.IGNORECASE
    )
    
    sol_pat = re.compile(
        r"(?:^|\n|\s+)\*?\*?(?:Answer|Ans(?:wer)?|Solution|Explanation|Description|Sol|Exp|Hint|വിശദീകരണം|പരിഹാരം|വിവരണം|സൂചന)[\*\:\.\-\s]+",
        re.IGNORECASE
    )

    ans_m = ans_pat.search(clean_text)
    sol_m = sol_pat.search(clean_text)

    if ans_m and sol_m:
        # We found "Answer: A" or similar
        marked_answer = ans_m.group(1).upper()
        # The explanation should be everything after "Answer:" (which is sol_m.end())
        # So it preserves the "A" or "(a)" if it's part of the text.
        explanation = clean_text[sol_m.end():].strip()
        clean_text = clean_text[:sol_m.start()].strip()
    elif sol_m:
        # Found "Solution:" or "Answer: The equipotential..." (no option letter)
        explanation = clean_text[sol_m.end():].strip()
        clean_text = clean_text[:sol_m.start()].strip()
        ans_in_sol = re.search(r"(?:Option\s*[:\-]*\s*\(?([A-D])\)?\s*is\s*correct|\(?([A-D])\)?\s*is\s*(?:the\s*)?correct\s*(?:option|answer))", explanation, re.IGNORECASE)
        if ans_in_sol:
            marked_answer = (ans_in_sol.group(1) or ans_in_sol.group(2)).upper()

    return clean_text, marked_answer, explanation

texts = [
    "Answer: \n(a) Equipotential surfaces of the system; \n(b) Equipotential surface get closer...",
    "Answer: C\nExplanation: Because it is C.",
    "Answer: B\nIt is B because of gravity.",
    "Solution: Option A is correct since...",
    "Answer: The equipotential surfaces are...",
    "Correct Answer: D"
]

for t in texts:
    c, a, e = parse_trailing_meta(t)
    print("---")
    print(f"TEXT: {repr(t)}")
    print(f"Clean: {repr(c)}")
    print(f"Ans: {repr(a)}")
    print(f"Exp: {repr(e)}")

