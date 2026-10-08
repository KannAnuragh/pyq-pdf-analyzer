import os
import re

def patch_file(path):
    with open(path, "r") as f:
        content = f.read()

    # correction_node.py replacements
    content = content.replace(
        "sub_parts = extract_subparts_from_text(clean_stem)",
        "sub_parts, clean_stem = extract_subparts_from_text(clean_stem)"
    )
    content = content.replace(
        "extracted_sub = extract_subparts_from_text(q.get(\"stem\", \"\"))",
        "extracted_sub, clean_stem_ret = extract_subparts_from_text(q.get(\"stem\", \"\"))\n                    if extracted_sub:\n                        q[\"stem\"] = clean_stem_ret"
    )
    # Another pattern in correction_node.py
    content = content.replace(
        "extracted_sub = extract_subparts_from_text(clean_stem)",
        "extracted_sub, clean_stem = extract_subparts_from_text(clean_stem)"
    )

    # block_classifier.py replacements
    content = content.replace(
        "extracted_sub = extract_subparts_from_text(stem)",
        "extracted_sub, stem = extract_subparts_from_text(stem)"
    )
    
    # test_suite.py replacements
    content = content.replace(
        "sub_parts = extract_subparts_from_text(text)",
        "sub_parts, _ = extract_subparts_from_text(text)"
    )
    content = content.replace(
        "det_sub = extract_subparts_from_text(",
        "det_sub, _ = extract_subparts_from_text("
    )

    with open(path, "w") as f:
        f.write(content)

patch_file("graph/correction_node.py")
patch_file("ingest/block_classifier.py")
patch_file("scratch/test_suite.py")

print("Patched usages.")
