import pymupdf
doc = pymupdf.open('/data/uploads/dpp/12CB CH-7 Alternating Current (DPP).pdf')
blocks = doc[1].get_text("blocks")
for b in blocks:
    print(f"y0: {b[1]:.2f}, text: {repr(b[4].strip() if b[6] == 0 else 'IMAGE')[:30]}")
