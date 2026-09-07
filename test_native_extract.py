import sys; sys.path.append('.')
import re
import pymupdf
from ingest.font_mapper import get_or_create_font_map

def convert_visual_to_logical(text: str) -> str:
    # Phase 3: Re-order pre-base vowel matras (െ, േ, ൈ) to post-base position
    text = re.sub(r'([െേൈ])([ക-ഹാ-ൌ്യ്ര്വ]+)', r'\2\1', text)
    return text

doc = pymupdf.open('/home/anuragh/Downloads/144-2026 M_removed.pdf')
page_idx = 0
font_map = get_or_create_font_map(doc, page_idx)

# Manually add the missing ligatures for UXAVQA+Rachana-Regular
# CID 5 is ൻറ (nt)
# Let's map any unknown CIDs that we can guess.
if 'f-2-1' in font_map:
    font_map['f-2-1']["5"] = "ൻറ"
    font_map['f-2-1']["13"] = "ങ്ങ"
    font_map['f-2-1']["14"] = "ച്ച"
    font_map['f-2-1']["16"] = "ത്ത"
    font_map['f-2-1']["17"] = "ന്ന"
    font_map['f-2-1']["22"] = "മ്പ"
    font_map['f-2-1']["25"] = "റ്റ"

page = doc[page_idx]
page_text_blocks = []
rawdict = page.get_text("rawdict")
for b in rawdict.get("blocks", []):
    if "lines" not in b: continue
    block_text = []
    for l in b["lines"]:
        line_text = []
        for s in l["spans"]:
            span_font = s["font"]
            span_text = ""
            for c in s["chars"]:
                char_str = c["c"]
                char_code = ord(char_str) if len(char_str) == 1 else 0
                # Using the stripped ToUnicode GID mapped directly
                if span_font in font_map and str(char_code) in font_map[span_font]:
                    span_text += font_map[span_font][str(char_code)]
                else:
                    span_text += char_str
            line_text.append(span_text)
        block_text.append(" ".join(line_text))
    page_text_blocks.append("\n".join(block_text))
page_text = "\n\n".join(page_text_blocks)

print("--- BEFORE REORDERING ---")
print(page_text[:500])

page_text = convert_visual_to_logical(page_text)

print("\n--- AFTER REORDERING ---")
print(page_text[:500])

