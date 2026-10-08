import re

def strip_subparts_text_from_stem(stem: str, sub_parts: dict) -> str:
    if not stem or not sub_parts:
        return stem
    
    # Sort subparts by length (longest first) to avoid partial replacements
    # Actually, if the stem ends with the subparts text, we should just truncate.
    # It's safer to just do a simple replacement if the subpart text is substantial.
    
    clean_stem = stem
    for k, v in sub_parts.items():
        # Only replace if the text is long enough to avoid accidental matches
        if len(v) > 10:
            # Escape regex chars in v
            # Make whitespace flexible
            v_pat = re.escape(v.strip())
            v_pat = re.sub(r'\\\s+', r'\\s+', v_pat)
            
            # Replace the subpart text in the stem (ignoring case)
            clean_stem = re.sub(v_pat, '', clean_stem, flags=re.IGNORECASE)
            
            # Also try to remove any orphaned markers like (a), a., etc. that might be left behind
            marker_pat = r'(?:^|\n|\s+)\(?' + re.escape(k) + r'\)?[\.\:]?\s*$'
            clean_stem = re.sub(marker_pat, '', clean_stem, flags=re.IGNORECASE)
            
    # Clean up trailing newlines and spaces
    clean_stem = clean_stem.strip()
    return clean_stem

stem = "Explain how electron mobility changes for a good conductor, when:\nThe temperature of the conductor decreases.\nThe applied potential difference is increased."
sub = {
  "A": "The temperature of the conductor decreases.",
  "B": "The applied potential difference is increased."
}

print(strip_subparts_text_from_stem(stem, sub))
