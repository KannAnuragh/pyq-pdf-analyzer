import pymupdf
doc = pymupdf.open('/home/anuragh/Downloads/144-2026 M_removed.pdf')
page = doc[0]
rawdict = page.get_text("rawdict")

for b in rawdict.get("blocks", []):
    if "lines" not in b: continue
    for l in b["lines"]:
        for s in l["spans"]:
            if "f-2-1" in s["font"]:
                for c in s["chars"]:
                    char_str = c["c"]
                    char_code = ord(char_str) if len(char_str) == 1 else 0
                    if char_str in ['*', '.', '/', '0']:
                        print(f"Found '{char_str}' (CID {char_code}) at {c['bbox']}")
