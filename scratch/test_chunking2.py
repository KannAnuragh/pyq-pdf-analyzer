import re

def parse_questions_from_text_new(raw_text: str):
    # Check for explicit 'Question X:' or 'Q X:'
    explicit_matches = list(re.finditer(r'(?i)(?:^|\n|\s+)\s*[\*\_]*(?:Question\s*|Q\.?\s*)[\*\_]*(\d+)\s*[\*\_]*(?::\-?|[\.\):-])\s*[\*\_]*', raw_text))
    
    if len(explicit_matches) >= 3:
        # If it explicitly says "Question 1:", we don't need strict empty line prefixes
        q_prefix_pattern = re.compile(r'(?:^|\n)\s*[\*\_]*(?:Question\s*|Q\.?\s*)[\*\_]*(\d+)\s*[\*\_]*(?::\-?|[\.\):-])\s*[\*\_]*', re.IGNORECASE)
    else:
        # Require an empty line before the question number to avoid inline numbered lists
        q_prefix_pattern = re.compile(r'(?:^|\n\s*\n)\s*[\*\_]*(\d+)[\*\_]*\s*(?:[\.\)]|:-)\s*[\*\_]*(?=\s)', re.IGNORECASE)
        
    matches = list(q_prefix_pattern.finditer(raw_text))
    
    valid_matches = []
    expected_next = 1
    for m in matches:
        q_num = int(m.group(1))
        if q_num >= expected_next or (q_num == 1 and expected_next > 5):
            valid_matches.append(m)
            expected_next = q_num + 1 if q_num != 1 else 2

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
