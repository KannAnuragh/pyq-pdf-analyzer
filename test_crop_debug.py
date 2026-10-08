import sys
import fitz
import re

def test(pdf_path, q_num, next_q_num):
    doc = fitz.open(pdf_path)
    page = doc[0] # assuming page 0 for test
    
    blocks = page.get_text("dict")["blocks"]
    start_y = None
    end_y = None
    
    pat_prefix = r'(?i)(?:^|\n)\s*[\*\_]*(?:Question\s*(?:No\.?)?|Q\.?\s*(?:No\.?)?|Sl\.?\s*No\.?|Problem\s*|Prob\.?\s*|ചോദ്യം\s*|പ്രശ്നം\s*)?[\(\[]?\s*'
    pat_suffix = r'\s*[\)\]]?\s*[\*\_]*(?::\-?|[\.\):-])(?:\s+|$|(?=[\(\[]))'
    
    q_start_pat = re.compile(f"{pat_prefix}{q_num}{pat_suffix}")
    next_q_pat = re.compile(f"{pat_prefix}{next_q_num}{pat_suffix}") if next_q_num else None
    
    found_start = False
    valid_y1s = []
    
    for b in blocks:
        if b["type"] == 0:
            for l in b["lines"]:
                txt = "".join([s["text"] for s in l["spans"]]).strip()
                if not txt: continue
                
                print(f"[{txt}] y0={l['bbox'][1]:.1f}, y1={l['bbox'][3]:.1f}")
                
                if not found_start:
                    if q_start_pat.search(txt):
                        print(">>> START MATCHED")
                        found_start = True
                        start_y = max(0, l["bbox"][1] - 12)
                        valid_y1s.append(l["bbox"][3])
                else:
                    if next_q_pat and next_q_pat.search(txt):
                        print(">>> NEXT Q MATCHED, BREAKING")
                        end_y = l["bbox"][1] - 5
                        break
                    else:
                        if l["bbox"][1] > page.rect.height - 70 and re.match(r'^(?:page\s*)?\d+(?:\s*of\s*\d+)?$', txt, re.IGNORECASE):
                            continue
                        valid_y1s.append(l["bbox"][3])
            
            if next_q_pat and end_y is not None:
                break
        else:
            if found_start:
                valid_y1s.append(b["bbox"][3])
                print(f"[IMAGE] y1={b['bbox'][3]}")

test("/home/anuragh/Downloads/dpp/dpp/ch7.pdf", 1, 2)
