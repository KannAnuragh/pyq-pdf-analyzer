import re

def parse_trailing_meta(text: str):
    clean_text = text.strip()
    marked_answer = None
    explanation = None

    ans_pat = re.compile(
        r"(?:^|\n|\s+)(?:Answer|Ans(?:wer)?|Correct(?:\s*Answer)?|ഉത്തരം)\s*[:\-]*\s*(?:is\s*)?(?:Option\s*[:\-]*\s*)?(?:\()?([A-D])(?:\))?\b",
        re.IGNORECASE
    )
    sol_pat = re.compile(
        r"(?:^|\n|\s+)(?:Solution|Explanation|Description|Sol|Exp|Hint|വിശദീകരണം|പരിഹാരം|വിവരണം|സൂചന)\s*[:\.\-]+\s*",
        re.IGNORECASE
    )

    ans_m = ans_pat.search(clean_text)
    sol_m = sol_pat.search(clean_text)

    if ans_m and sol_m:
        if ans_m.start() < sol_m.start():
            marked_answer = ans_m.group(1).upper()
            explanation = clean_text[sol_m.end():].strip()
            clean_text = clean_text[:ans_m.start()].strip()
        else:
            marked_answer = ans_m.group(1).upper()
            explanation = (clean_text[sol_m.end():ans_m.start()] + "\n" + clean_text[ans_m.end():]).strip()
            clean_text = clean_text[:sol_m.start()].strip()
    elif ans_m:
        marked_answer = ans_m.group(1).upper()
        trailing = clean_text[ans_m.end():].strip()
        trailing_sol_m = sol_pat.search(trailing)
        if trailing_sol_m:
            explanation = trailing[trailing_sol_m.end():].strip()
        elif trailing and (trailing.lower().startswith("because") or trailing.lower().startswith("since")):
            explanation = trailing
        clean_text = clean_text[:ans_m.start()].strip()
    elif sol_m:
        explanation = clean_text[sol_m.end():].strip()
        clean_text = clean_text[:sol_m.start()].strip()

    return clean_text, marked_answer, explanation

def separate_stem_and_options(content: str):
    opt_pattern = re.compile(r'(?:^|\n|\s+)(?:\()?([A-D])(?:\)|\.|:-)\s*', re.IGNORECASE)
    matches = list(opt_pattern.finditer(content))
    
    best_sequence = []
    for i, m in enumerate(matches):
        curr_seq = [m]
        curr_letter = m.group(1).upper()
        for next_m in matches[i+1:]:
            next_letter = next_m.group(1).upper()
            if next_letter > curr_letter:
                curr_seq.append(next_m)
                curr_letter = next_letter
        if len(curr_seq) > len(best_sequence):
            best_sequence = curr_seq

    stem = content.strip()
    options = {}
    marked_answer = None
    explanation = None

    if len(best_sequence) >= 2:
        stem = content[:best_sequence[0].start()].strip()
        for idx, m in enumerate(best_sequence):
            letter = m.group(1).upper()
            start_pos = m.end()
            end_pos = best_sequence[idx+1].start() if idx < len(best_sequence) - 1 else len(content)
            opt_text = content[start_pos:end_pos].strip()
            
            clean_opt, ans, exp = parse_trailing_meta(opt_text)
            if ans and not marked_answer:
                marked_answer = ans
            if exp and not explanation:
                explanation = exp
            options[letter] = clean_opt
    
    return stem, options, marked_answer, explanation

text = """1. Three charges q, −q and q0 are placed as shown in figure. The magnitude of the 
net force on the charge q0 at point O is [k =
1
4πε0]  
 
 
[1 mark] 
 
(a) 0 
(b) 
2kqq0
a2 
(c) 
√2kqq0
a2
 
(d) 
1
√2 
kqq0
a2 
Answer: 
(c) 
√2kqq0
a2"""

stem, options, marked_ans, explanation = separate_stem_and_options(text)
import json
print(json.dumps({"stem": stem, "options": options, "marked_ans": marked_ans, "explanation": explanation}, indent=2))
