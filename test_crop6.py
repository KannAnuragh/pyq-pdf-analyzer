import sys
import re
import pymupdf

def get_question_crop_rect(page, question_number: int, next_question_number: int = None):
    blocks = page.get_text("blocks")
    start_y = None
    end_y = None
    pat_prefix = r'(?i)(?:^|\n)\s*[\*\_]*(?:Question\s*(?:No\.?)?|Q\.?\s*(?:No\.?)?|Sl\.?\s*No\.?|Problem\s*|Prob\.?\s*|ചോദ്യം\s*|പ്രശ്നം\s*)?[\(\[]?\s*'
    pat_suffix = r'\s*[\)\]]?\s*[\*\_]*(?::\-?|[\.\):-])(?:\s+|$|(?=[\(\[]))'
    q_start_pat = re.compile(f"{pat_prefix}{question_number}{pat_suffix}")
    next_q_pat = re.compile(f"{pat_prefix}{next_question_number}{pat_suffix}") if next_question_number else None
    question_blocks = []
    found_start = False
    
    for b in blocks:
        is_text = b[6] == 0
        txt = b[4] if is_text else ""
        if not found_start:
            if is_text and q_start_pat.search(txt):
                found_start = True
                start_y = max(0, b[1] - 12)
                question_blocks.append(b)
        else:
            if next_q_pat and is_text and next_q_pat.search(txt):
                end_y = b[1] - 5
                break
            else:
                question_blocks.append(b)
                
    if start_y is not None:
        if end_y is None:
            if question_blocks:
                valid_y1s = []
                for b in question_blocks:
                    if b[1] > page.rect.height - 70:
                        continue
                    valid_y1s.append(b[3])
                if valid_y1s:
                    end_y = max(valid_y1s) + 15
                else:
                    end_y = max(start_y + 100, page.rect.height - 30)
            else:
                end_y = max(start_y + 100, page.rect.height - 30)
        end_y = min(end_y, page.rect.height)
        return pymupdf.Rect(0, start_y, page.rect.width, end_y)
    return None

doc = pymupdf.open('/data/uploads/dpp/12CB CH-7 Alternating Current (DPP).pdf')
page = doc[1]

rect6 = get_question_crop_rect(page, 6, None)
print("Q6 Rect:", rect6)

blocks = page.get_text("blocks")
for b in blocks:
    is_text = b[6] == 0
    txt = b[4] if is_text else ""
    if rect6 and rect6.y0 <= b[1] <= rect6.y1:
        print(f"IN Q6 CROP [y0={b[1]} y1={b[3]}]:", repr(txt.strip()))
