import pymupdf
doc = pymupdf.open("dummy.pdf")
page = doc[0]
print(hasattr(page, "get_drawings"))
