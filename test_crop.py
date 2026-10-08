import sys
import json
import pymupdf

sys.path.insert(0, '/app/pdf_qa_agent')
from api.routes import get_question_crop_rect

doc = pymupdf.open('data/uploads/dpp/12CB CH-7 Alternating Current (DPP).pdf')
page = doc[0]

# test crop for Q4 (next is 5)
rect4 = get_question_crop_rect(page, 4, 5)
print("Q4 Rect:", rect4)

# test crop for Q5 (next is 6)
rect5 = get_question_crop_rect(page, 5, 6)
print("Q5 Rect:", rect5)

# print blocks between Q4 and Q5
blocks = page.get_text("blocks")
for b in blocks:
    is_text = b[6] == 0
    txt = b[4] if is_text else ""
    if rect4 and rect4.y0 <= b[1] <= rect4.y1:
        print("IN Q4 CROP:", txt.strip())
