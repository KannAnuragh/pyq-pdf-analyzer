import pymupdf
import re

def get_question_crop_rect(page, question_number: int, next_question_number: int = None):
    blocks = page.get_text("dict")["blocks"]
    start_y = None
    end_y = None
    
    pat_prefix = r'(?i)(?:^|\n)\s*[\*\_]*(?:Question\s*(?:No\.?)?|Q\.?\s*(?:No\.?)?|Sl\.?\s*No\.?|Problem\s*|Prob\.?\s*|ചോദ്യം\s*|പ്രശ്നം\s*)?[\(\[]?\s*'
    pat_suffix = r'\s*[\)\]]?\s*[\*\_]*(?::\-?|[\.\):-])(?:\s+|$|(?=[\(\[]))'
    
    q_start_pat = re.compile(f"{pat_prefix}{question_number}{pat_suffix}")
    next_q_pat = re.compile(f"{pat_prefix}{next_question_number}{pat_suffix}") if next_question_number else None
    
    found_start = False
    valid_y1s = []
    
    for b in blocks:
        if b["type"] == 0:
            for l in b["lines"]:
                txt = "".join([s["text"] for s in l["spans"]]).strip()
                if not txt: continue
                
                if not found_start:
                    if q_start_pat.search(txt):
                        found_start = True
                        start_y = max(0, l["bbox"][1] - 12)
                        valid_y1s.append(l["bbox"][3])
                else:
                    if next_q_pat and next_q_pat.search(txt):
                        end_y = l["bbox"][1] - 5
                        break
                    else:
                        valid_y1s.append(l["bbox"][3])
            
            if next_q_pat and end_y is not None:
                break
        else:
            if found_start:
                valid_y1s.append(b["bbox"][3])
                
    if start_y is not None:
        if end_y is None:
            if valid_y1s:
                end_y = max(valid_y1s) + 15
            else:
                end_y = max(start_y + 100, page.rect.height - 30)
        end_y = min(end_y, page.rect.height)
        return pymupdf.Rect(0, start_y, page.rect.width, end_y)
    return None

doc = pymupdf.open('/tmp/ch1.pdf')
page = doc[1]
print("Q4 Rect:", get_question_crop_rect(page, 4, 5))
