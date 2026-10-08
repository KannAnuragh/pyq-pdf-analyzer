import pymupdf
doc = pymupdf.open('/tmp/ch1.pdf')
page = doc[1]
for b in page.get_text("dict")["blocks"]:
    if b["type"] == 0:
        for l in b["lines"]:
            text = "".join([s["text"] for s in l["spans"]])
            print(f"y0: {l['bbox'][1]:.2f}, y1: {l['bbox'][3]:.2f}, text: {repr(text)}")
