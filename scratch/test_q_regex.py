import re

text = """**7.** What is the work done...
**Answer:**
Some explanation.

**8.** X and Y are two equipotential surfaces..."""

q_prefix_pattern = r'[\*\_]*\d+[\*\_]*\s*(?:[\.\)]|:-)\s*[\*\_]*\s*'
q_extract_pattern = r'^[\*\_]*(\d+)[\*\_]*\s*(?:[\.\)]|:-)\s*[\*\_]*\s*(.*)'

blocks = re.split(fr'\n(?={q_prefix_pattern})|^(?={q_prefix_pattern})', text, flags=re.MULTILINE | re.IGNORECASE)

for i, block in enumerate(blocks):
    if not block: continue
    print(f"--- Block {i} ---")
    print(block.strip())
    
    q_match = re.match(q_extract_pattern, block.strip(), flags=re.DOTALL | re.IGNORECASE)
    if q_match:
        print(f"Num: {q_match.group(1)}")
