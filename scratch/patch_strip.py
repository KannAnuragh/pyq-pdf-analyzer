import os

path = "ingest/option_answer_separator.py"
with open(path, "r") as f:
    content = f.read()

new_func = """
def strip_subparts_text_from_stem(stem: str, sub_parts: Dict[str, str]) -> str:
    \"\"\"
    Removes the text of subparts from the stem if they were duplicated by the LLM
    without their list markers (e.g., (a), (b)).
    \"\"\"
    if not stem or not sub_parts:
        return stem
        
    clean_stem = stem
    for k, v in sub_parts.items():
        if len(v) > 10:
            import re
            v_pat = re.escape(v.strip())
            v_pat = re.sub(r'\\\\s\+', r'\\\\s+', v_pat)
            clean_stem = re.sub(v_pat, '', clean_stem, flags=re.IGNORECASE)
            
    return clean_stem.strip()
"""

if "def strip_subparts_text_from_stem" not in content:
    content += "\n" + new_func

with open(path, "w") as f:
    f.write(content)

print("Added strip_subparts_text_from_stem")
