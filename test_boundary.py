import re
import sys
import os

from ingest.layout_segmenter import _is_question_boundary, QuestionSegment, LayoutElement

elem = LayoutElement(
    type="text",
    text="3. Loop Q carries a current 9 A in the anticlockwise direction.",
    bbox=[100, 200, 500, 215],
    page=1,
    font_size=12.0,
    is_bold=False,
    x_indent=100/800.0,
    needs_structural_ocr=False,
    region_type=""
)

prev_elem = LayoutElement(
    type="text",
    text="Two concentric and coplanar circular loops P and Q have their radii in the ratio 2 :",
    bbox=[100, 180, 500, 195],
    page=1,
    font_size=12.0,
    is_bold=False,
    x_indent=100/800.0,
    needs_structural_ocr=False,
    region_type=""
)

curr_seg = QuestionSegment(q_num=2, page=1, elements=[prev_elem], section=None)

body_size = 12.0
gap_size = 1

res = _is_question_boundary(elem, body_size, gap_size, prev_elem, curr_seg)
print("Result with normal y_gap:", res)

# test with large y_gap
elem.bbox = [100, 230, 500, 245]
res2 = _is_question_boundary(elem, body_size, gap_size, prev_elem, curr_seg)
print("Result with larger y_gap (35px):", res2)

# test if bold
elem.is_bold = True
res3 = _is_question_boundary(elem, body_size, gap_size, prev_elem, curr_seg)
print("Result if bold:", res3)
