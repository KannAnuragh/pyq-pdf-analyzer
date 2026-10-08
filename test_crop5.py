import pymupdf
doc = pymupdf.open('/data/uploads/dpp/12CB CH-7 Alternating Current (DPP).pdf')
page = doc[1]
blocks = page.get_text("blocks")
for b in blocks:
    if b[6] == 0:
        txt = b[4].strip()
        if txt.startswith('7.'):
            print(f"Found Q7 on page 1")
