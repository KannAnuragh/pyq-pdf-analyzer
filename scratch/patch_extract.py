import os
import re

def update_separator():
    path = "ingest/option_answer_separator.py"
    with open(path, "r") as f:
        content = f.read()

    # Update signature
    content = content.replace("def extract_subparts_from_text(text: str) -> Optional[Dict[str, str]]:", "def extract_subparts_from_text(text: str) -> Tuple[Optional[Dict[str, str]], str]:")
    
    # Update returns
    content = content.replace("return None\n", "return None, text\n")
    
    # In extract_subparts_from_text, update the final return
    # Find the end of the function
    find_str = """
    if not is_valid_subparts(subparts):
        return None, text

    return subparts"""
    
    replace_str = """
    if not is_valid_subparts(subparts):
        return None, text

    first_match_start = valid_matches[0][1].start()
    clean_stem = text[:first_match_start].strip()

    return subparts, clean_stem"""
    
    if find_str in content:
        content = content.replace(find_str, replace_str)
    
    with open(path, "w") as f:
        f.write(content)

update_separator()
print("Updated option_answer_separator.py")
