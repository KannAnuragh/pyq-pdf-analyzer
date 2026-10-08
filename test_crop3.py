import pymupdf
import re
doc = pymupdf.open('/data/uploads/dpp/12CB CH-7 Alternating Current (DPP).pdf')
page = doc[1]
blocks = page.get_text("blocks")
for b in blocks:
    if b[6] == 0:
        print(repr(b[4].strip()))
