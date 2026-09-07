import sys; sys.path.append('.')
import pymupdf
from ingest.font_mapper import get_or_create_font_map

doc = pymupdf.open('/home/anuragh/Downloads/144-2026 M_removed.pdf')
font_map = get_or_create_font_map(doc, 0)
fmap = font_map.get('f-2-1', {})

unmapped_counts = {}

for page_idx in range(min(5, len(doc))):
    page = doc[page_idx]
    rawdict = page.get_text("rawdict")
    for b in rawdict.get("blocks", []):
        if "lines" not in b: continue
        for l in b["lines"]:
            for s in l["spans"]:
                if 'Rachana' in s["font"]:
                    for c in s["chars"]:
                        cid = str(ord(c["c"]))
                        if cid not in fmap:
                            unmapped_counts[cid] = unmapped_counts.get(cid, 0) + 1

for cid, count in sorted(unmapped_counts.items(), key=lambda x: -x[1]):
    print(f"CID {cid}: {count} occurrences")
