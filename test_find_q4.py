import pymupdf
import glob
for pdf in glob.glob('/data/uploads/dpp/*.pdf'):
    doc = pymupdf.open(pdf)
    for p_num, page in enumerate(doc):
        text = page.get_text("text").replace('\n', ' ')
        if "parallel" in text and "sheets" in text and "surface densities" in text:
            print(f"FOUND IN: {pdf} Page {p_num}")
