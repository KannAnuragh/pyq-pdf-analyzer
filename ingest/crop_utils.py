import re
import pymupdf
from typing import Optional, List, Dict, Any, Tuple

STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'on', 'at', 'by', 'for', 'with',
    'about', 'between', 'into', 'through', 'before', 'after', 'above', 'below', 'from',
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
    'do', 'does', 'did', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'its', 'it', 'if', 'then', 'so', 'than', 'such', 'both', 'all', 'any'
}

def clean_anchor_words(text: str) -> List[str]:
    """Extracts significant alphanumeric words (excluding stop words) for anchor matching."""
    if not text:
        return []
    words = re.findall(r'[a-zA-Z0-9\u0d00-\u0d7f\u0900-\u097f]+', text.lower())
    return [w for w in words if (len(w) >= 2 or w in ['x', 'y', 'z']) and w not in STOP_WORDS]

def is_option_line(text: str) -> bool:
    """Detects if a text line is an option row (e.g. 1) A 2) B, or 1) Magnified, or (1) Option, or A) 6)."""
    if not text:
        return False
    s = text.strip()
    # Multiple options on the same line: e.g. "1) ... 2) ..." or "(1) ... (2) ..."
    if len(re.findall(r'(?:^|\s)(?:[1-4]|[A-Da-d])\)', s)) >= 2:
        return True
    if len(re.findall(r'\((?:[1-4]|[A-Da-d])\)', s)) >= 2:
        return True
    # Single option starting with numbered or lettered parenthesis: "1) ...", "(1) ...", "A) ...", "(A) ..."
    if re.match(r'^\s*[1-4]\)\s+', s) or re.match(r'^\s*\([1-4]\)\s+', s):
        return True
    if re.match(r'^\s*[A-Da-d]\)\s+', s) or re.match(r'^\s*\([A-Da-d]\)\s+', s):
        return True
    return False

def is_line_q_marker(line_text: str, q_num: int, expected_style: Optional[str] = None) -> Tuple[bool, float, Optional[str]]:
    """
    Checks if line_text starts with question number q_num.
    Returns (is_match, confidence_score, style).
    Styles: 'prefix', 'dot', 'bracket', 'paren', 'standalone'
    Distinguishes question markers from numbered options like '1)', '2)', '3)', '4)'.
    """
    if not line_text or q_num is None:
        return False, 0.0, None
        
    s = line_text.strip()
    qn_str = str(q_num)
    
    # 1. Explicit prefix: Question 1, Q1, Q.1, Problem 1, ചോദ്യം 1
    if re.match(rf'(?i)^\s*(?:Question\s*(?:No\.?)?|Q\.?\s*(?:No\.?)?|Sl\.?\s*No\.?|Problem\s*|Prob\.?\s*|Que\b|Qn\b|ചോദ്യം\s*|പ്രശ്നം\s*)\s*[\(\[]?\s*{qn_str}\b', s):
        return True, 1.0, "prefix"
        
    # 2. Both brackets: (1), [1]
    if re.match(rf'^\s*[\(\[]{qn_str}[\)\]]', s):
        # In papers with numbered options, (1)-(4) on option-like lines are options
        if is_option_line(s) and q_num <= 4 and expected_style != "bracket":
            return False, 0.0, None
        return True, 0.90, "bracket"
        
    # 3. Number + dot / colon / dash: 1., 1:-, 1-, 1.7th, 6.HCF, 3. .
    m = re.match(rf'^\s*[\*\_]*{qn_str}\s*(?:[\.\:\-]+|\s*\.\s*\.)\s*(.*)', s)
    if m:
        rest = m.group(1).strip()
        # Discard pure floats like 1.5 kg, but keep ordinals like 1.7th or words like 6.HCF
        if re.match(r'^\d+(?:\.\d+)?\s*(?:cm|m|kg|s|v|w|hz|%|hours|years)?\b', rest) and not re.match(r'^\d+(?:st|nd|rd|th)\b', rest):
            return False, 0.0, None
        return True, 0.95, "dot"
        
    # 4. Number + lone closing bracket: 1), 2), 3), 4)
    # Lone closing parenthesis without open bracket for 1-4 is almost always an option marker
    if re.match(rf'^\s*[\*\_]*{qn_str}\s*\)\s*(.*)', s):
        if expected_style != "paren" and (is_option_line(s) or q_num <= 4):
            return False, 0.0, None
        return True, 0.50, "paren"
        
    # 5. Standalone number on line
    if re.match(rf'^\s*[\*\_]*{qn_str}\s*[\.\:\-]?\s*$', s):
        return True, 0.85, "standalone"
        
    return False, 0.0, None

def matches_stem(candidate_text: str, stem: str) -> bool:
    """Checks if candidate text matches stem using significant word overlap."""
    if not candidate_text or not stem:
        return False
    cand_words = clean_anchor_words(candidate_text)
    stem_words = clean_anchor_words(stem)
    if not stem_words or not cand_words:
        return False
    check_stem = stem_words[:6]
    matches = sum(1 for w in check_stem if w in cand_words)
    return matches >= min(2, len(check_stem))

def is_cover_or_instruction_page(page_text: str) -> bool:
    """Identifies cover, instructions, or syllabus pages that are not part of the question paper."""
    if re.search(r'(?i)(?:^|\n)\s*(?:Question\s*|Q\.?\s*|Sl\.?\s*No\.?)?1[\.\):-]\s+.*?(?:^|\n|\s+)(?:\(?[A-D][\.\):-]|\([a-d]\))', page_text, re.DOTALL):
        return False

    has_inst = any(w in page_text for w in [
        'j]¡SÇw°¥', 'നിർദ്ദേശങ്ങൾ', 'Instructions', 'INSTRUCTIONS',
        'DSh|LYL¡À]W¥¨Oç', 'ഉദ്യോഗാർത്ഥികൾക്കുള്ള', 'CANDIDATES'
    ])
    has_booklet = any(w in page_text for w in [
        'S\\Lh|kOñW', 'ചോദ്യപുസ്തക', 'Question Booklet', 'Alpha Code', 'AƒqSWLcV',
        'Max. Marks', 'Maximum Marks', 'yopU :'
    ])
    return has_inst and has_booklet

def find_question_page(
    doc: pymupdf.Document,
    question_number: int,
    hint_page: Optional[int] = None,
    question_stem: Optional[str] = None
) -> Optional[int]:
    """Finds physical page containing question_number, skipping cover/instruction pages."""
    # Fast path: hint_page check
    if hint_page is not None and 0 <= hint_page < len(doc):
        p_txt = doc[hint_page].get_text()
        if not is_cover_or_instruction_page(p_txt):
            if any(is_line_q_marker(line, question_number)[0] for line in p_txt.splitlines()):
                return hint_page
            if question_stem and matches_stem(p_txt, question_stem):
                return hint_page

    # Priority 1: Stem match across pages
    if question_stem:
        for p in range(len(doc)):
            p_txt = doc[p].get_text()
            if is_cover_or_instruction_page(p_txt):
                continue
            if matches_stem(p_txt, question_stem):
                return p

    # Priority 2: Line question marker match across pages
    for p in range(len(doc)):
        p_txt = doc[p].get_text()
        if is_cover_or_instruction_page(p_txt):
            continue
        if any(is_line_q_marker(line, question_number)[0] for line in p_txt.splitlines()):
            return p

    # Fallback: hint_page if in bounds
    if hint_page is not None and 0 <= hint_page < len(doc):
        return hint_page

    return None

def get_question_crop_rect(
    page: pymupdf.Page,
    question_number: int,
    next_question_number: Optional[int] = None,
    next_question_stem: Optional[str] = None,
    is_continuation: bool = False,
    current_question_stem: Optional[str] = None,
    current_options: Optional[Any] = None,
) -> Optional[pymupdf.Rect]:
    """Calculates tight bounding box Rect for the specified question on page."""
    page_data = page.get_text("dict")
    blocks = page_data["blocks"]
    page_w = page.rect.width
    page_h = page.rect.height
    
    # Collect all lines with bbox and text
    lines = []
    drawing_boxes = []
    for b in blocks:
        if b["type"] == 0:
            for l in b["lines"]:
                txt = "".join([s["text"] for s in l["spans"]]).strip()
                if txt:
                    lines.append({
                        "bbox": l["bbox"],
                        "text": txt,
                        "y0": l["bbox"][1],
                        "y1": l["bbox"][3],
                        "x0": l["bbox"][0],
                        "x1": l["bbox"][2],
                    })
        elif b["type"] == 1:
            drawing_boxes.append(b["bbox"])
            
    for d in page.get_drawings():
        r = d.get("rect")
        if r and r.width > 3 and r.height > 3:
            drawing_boxes.append((r.x0, r.y0, r.x1, r.y1))

    if not lines and not is_continuation:
        return None

    # Detect multi-column layout
    is_two_column = False
    left_col_count = sum(1 for l in lines if l["x1"] <= page_w * 0.52)
    right_col_count = sum(1 for l in lines if l["x0"] >= page_w * 0.48)
    if len(lines) >= 6 and left_col_count >= 3 and right_col_count >= 3:
        is_two_column = True

    # Sort lines vertically
    lines.sort(key=lambda x: (x["y0"], x["x0"]))
    
    start_y = 0.0 if is_continuation else None
    start_idx = -1
    detected_style = None
    
    # --- STEP 1: Find start_y ---
    if not is_continuation:
        best_match_idx = -1
        best_score = 0.0
        best_style = None
        
        for idx, l in enumerate(lines):
            # Ignore headers in top 40pt if text looks like header
            if l["y0"] < 40 and re.search(r'(?i)(?:examination|test|paper|class|max marks|time:)', l["text"]):
                continue
                
            is_match, score, style = is_line_q_marker(l["text"], question_number)
            if is_match:
                if current_question_stem:
                    nearby = " ".join([lines[j]["text"] for j in range(idx, min(len(lines), idx + 3))])
                    if matches_stem(nearby, current_question_stem):
                        score += 0.5
                if score > best_score:
                    best_score = score
                    best_match_idx = idx
                    best_style = style
                    
        # Strategy B: If no match found, use current_question_stem search
        if best_match_idx == -1 and current_question_stem:
            stem_anchor = clean_anchor_words(current_question_stem)[:5]
            if stem_anchor:
                for idx, l in enumerate(lines):
                    cand = clean_anchor_words(l["text"])
                    if sum(1 for w in stem_anchor if w in cand) >= min(2, len(stem_anchor)):
                        if idx > 0 and is_line_q_marker(lines[idx - 1]["text"], question_number)[0]:
                            best_match_idx = idx - 1
                            best_style = is_line_q_marker(lines[idx - 1]["text"], question_number)[2]
                        else:
                            best_match_idx = idx
                            best_style = "dot"
                        break
                        
        if best_match_idx != -1:
            start_idx = best_match_idx
            start_y = max(0.0, lines[best_match_idx]["y0"] - 10.0)
            detected_style = best_style
        else:
            return None
    else:
        # Continuation: start at top content line below header
        start_idx = 0
        first_content_lines = [l for l in lines if l["y0"] > 35]
        if first_content_lines:
            start_y = max(0.0, first_content_lines[0]["y0"] - 8.0)
            start_idx = lines.index(first_content_lines[0])
        elif lines:
            start_y = max(0.0, lines[0]["y0"] - 8.0)
            start_idx = 0
        else:
            return None

    # --- STEP 2: Find end_y ---
    end_y = None
    next_q_idx = -1
    
    if next_question_number is not None:
        for idx in range(start_idx + 1, len(lines)):
            l = lines[idx]
            
            # Skip option lines of the current question
            if is_option_line(l["text"]) and not re.search(rf'(?i)(?:Question|Q\.?|Sl\.?|Prob)\s*{next_question_number}\b', l["text"]):
                continue
                
            is_match, score, _ = is_line_q_marker(l["text"], next_question_number, expected_style=detected_style)
            if is_match:
                if next_question_stem:
                    nearby = " ".join([lines[j]["text"] for j in range(idx, min(len(lines), idx + 3))])
                    if matches_stem(nearby, next_question_stem) or len(clean_anchor_words(next_question_stem)) == 0:
                        next_q_idx = idx
                        break
                    else:
                        continue
                else:
                    next_q_idx = idx
                    break
            elif next_question_stem:
                cand = clean_anchor_words(l["text"])
                next_anchor = clean_anchor_words(next_question_stem)[:5]
                if next_anchor and sum(1 for w in next_anchor if w in cand) >= min(2, len(next_anchor)):
                    next_q_idx = idx
                    break

    if next_q_idx != -1:
        end_y = lines[next_q_idx]["y0"] - 5.0
        q_lines = lines[start_idx:next_q_idx]
    else:
        # Next question NOT on this page (or last question)
        q_lines = []
        for idx in range(start_idx, len(lines)):
            l = lines[idx]
            if re.match(r'(?i)^\s*(?:ANSWER\s*KEY|ANSWERS|ANSWER\s*SHEET)\b', l["text"]):
                break
            if l["y0"] > page_h - 70 and re.match(r'^(?:page\s*)?\d+(?:\s*(?:of|\/)\s*\d+)?$', l["text"], re.IGNORECASE):
                continue
            if l["y0"] > page_h - 60 and re.search(r'(?i)(?:space for rough work|turn over|p\.?t\.?o\.?)', l["text"]):
                break
            q_lines.append(l)
            
        if q_lines:
            end_y = max(l["y1"] for l in q_lines) + 12.0
        else:
            end_y = min(start_y + 150.0, page_h)

    # Include any drawings / images located between start_y and end_y
    for db in drawing_boxes:
        d_y0, d_y1 = db[1], db[3]
        if d_y0 >= start_y - 5.0 and d_y0 <= (end_y or page_h):
            if end_y is not None:
                end_y = max(end_y, d_y1 + 8.0)

    # Determine horizontal bounds
    crop_x0 = 0.0
    crop_x1 = page_w
    if is_two_column and q_lines:
        max_x1 = max(l["x1"] for l in q_lines)
        min_x0 = min(l["x0"] for l in q_lines)
        if max_x1 <= page_w * 0.52:
            # Confined to Left column
            crop_x0 = 0.0
            crop_x1 = min(page_w * 0.52, max_x1 + 15.0)
        elif min_x0 >= page_w * 0.48:
            # Confined to Right column
            crop_x0 = max(page_w * 0.48, min_x0 - 15.0)
            crop_x1 = page_w

    # Clamp
    end_y = min(end_y, page_h)
    start_y = max(0.0, start_y)
    
    if end_y <= start_y + 10.0:
        return None
        
    return pymupdf.Rect(crop_x0, start_y, crop_x1, end_y)
