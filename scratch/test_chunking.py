import re

def parse_questions_from_text_new(raw_text: str):
    extracted = []
    
    # Check for explicit 'Question X:' or 'Q X:'
    explicit_matches = list(re.finditer(r'(?i)(?:^|\n|\s+)\s*[\*\_]*(?:Question\s*|Q\.?\s*)[\*\_]*(\d+)\s*[\*\_]*(?::\-?|[\.\):-])\s*[\*\_]*', raw_text))
    
    if len(explicit_matches) >= 3:
        q_prefix_pattern = re.compile(r'(?:^|\n|\s+)\s*[\*\_]*(?:Question\s*|Q\.?\s*)[\*\_]*(\d+)\s*[\*\_]*(?::\-?|[\.\):-])\s*[\*\_]*', re.IGNORECASE)
    else:
        q_prefix_pattern = re.compile(r'(?:^|\n)\s*[\*\_]*(\d+)[\*\_]*\s*(?:[\.\)]|:-)\s*[\*\_]*(?=\s|$)', re.IGNORECASE)
        
    matches = list(q_prefix_pattern.finditer(raw_text))
    
    # Filter for monotonic sequence to avoid numbered lists in descriptions
    valid_matches = []
    last_q_num = 0
    for m in matches:
        q_num = int(m.group(1))
        # Allow resetting to 1 if we drop heavily (e.g. new section), but otherwise it must be > last_q_num
        if q_num == last_q_num + 1 or q_num == last_q_num + 2 or (q_num == 1 and last_q_num > 5) or (q_num > last_q_num):
            # Wait, if we are at Q45, and a list has 1, 2, 3...
            # if q_num == 1 and last_q_num > 5: allows section reset.
            # But what if we are at Q2, and the explanation has 1., 2.?
            # last_q_num is 2. Next match is 1. (q_num < last_q_num). We reject it.
            # Next match is 2. (q_num == last_q_num). We reject it.
            # Next match is 3. (q_num == last_q_num + 1). Wait! The list item 3 will be accepted!
            pass

    # A better heuristic for valid matches:
    # We collect all sequences of monotonically increasing numbers (with small gaps allowed)
    # The longest/dominant sequence is the question list. Or we just enforce strictly increasing 
    # numbers for the whole document, allowing one-time resets if it drops to exactly 1.
    
    # Actually, simpler logic:
    # A match is valid if it is strictly greater than the last accepted question number, 
    # OR if it's exactly 1 (signifying a new section reset). 
    # Wait, if we are at Q2, and a list item is "3.", it will be accepted!
    # How to prevent list items? Usually list items are indented, but OCR loses indent.
    # What if we ensure the block between questions is substantial, or just rely on the fallback?
    # Let's test the extraction with just strictly increasing logic.
    valid_matches = []
    expected_next = 1
    for m in matches:
        q_num = int(m.group(1))
        # If it's what we exactly expect, or slightly ahead (skipped question)
        if q_num >= expected_next:
            valid_matches.append(m)
            expected_next = q_num + 1
        elif q_num == 1 and expected_next > 5:
            # Section reset
            valid_matches.append(m)
            expected_next = 2

    # Now split the text based on valid_matches
    if not valid_matches:
        return []

    blocks = []
    for i, m in enumerate(valid_matches):
        start_idx = m.end()
        end_idx = valid_matches[i+1].start() if i + 1 < len(valid_matches) else len(raw_text)
        block_content = raw_text[start_idx:end_idx].strip()
        blocks.append({
            "question_number": int(m.group(1)),
            "content": block_content
        })

    return blocks

text = """
1. What is gravity?
It is a force.
Here are reasons:
1. Mass
2. Distance
3. Constant

2. Calculate force.
Options:
(A) 10
(B) 20
Answer: A
"""

res = parse_questions_from_text_new(text)
for r in res:
    print(f"Q{r['question_number']}: {repr(r['content'][:50])}...")
