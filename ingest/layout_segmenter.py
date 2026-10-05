"""
Layout-aware question segmenter.

Splits the span_blocks produced by router.extract_text_from_pdf into
QuestionSegment objects using ONLY structural cues:

  - x-indent (x0 / page_width, normalised to [0,1])
  - font size relative to the body average for this page
  - bold flag (PyMuPDF span flags bit 4)
  - a numbering pattern at the start of the span text

This replaces the three-regex (explicit_pat / paren_pat / num_pat) boundary
detection in parse_questions_from_text for PDFs that have a live text layer.
The key difference: a token is accepted as a question-boundary marker ONLY when
the structural cues corroborate the numbering pattern.  An abbreviation like
"A.C." at column-0 therefore does NOT trigger a false split because its font
size and indent are identical to body text, not to question headers.

The existing parse_questions_from_text() remains active as:
  a) the fallback path when no LLM is configured
  b) the path for scanned/OCR-only pages (span_blocks is empty)
  c) the correction_node rescan path
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LayoutElement:
    """A single rendered element from the span stream."""
    type: str           # "text" | "image" | "equation"
    text: str           # resolved text, or anchor token {{IMG:...}} / {{EQ:...}}
    bbox: list          # [x0, y0, x1, y1]
    page: int
    font_size: float
    is_bold: bool
    x_indent: float     # x0 / page_width, normalised to [0,1]
    needs_structural_ocr: bool = False
    region_type: str = ""


@dataclass
class QuestionSegment:
    """One question's worth of layout elements."""
    q_num: int
    page: int
    section: Optional[str]
    elements: list[LayoutElement] = field(default_factory=list)

    @property
    def flat_text(self) -> str:
        """Plain-text serialisation for the regex fallback path."""
        return " ".join(e.text for e in self.elements)

    def structured_text(self) -> str:
        """
        Layout-hinted serialisation for the LLM classifier prompt.

        Each element is prefixed with bracketed layout annotations:
          [BOLD][font=12.0][indent=0.02] 3. Which of the following...
          [font=11.0][indent=0.14] (A) Sodium chloride
          [font=11.0][indent=0.14] (B) Potassium nitrate
          [font=11.0][indent=0.02] Answer: B

        The LLM uses indent and bold as ground-truth structural signal
        instead of pattern-matching on flat text.
        """
        lines = []
        for e in self.elements:
            prefix = ""
            if e.is_bold:
                prefix += "[BOLD]"
            prefix += f"[font={e.font_size:.1f}][indent={e.x_indent:.2f}] "
            lines.append(prefix + e.text)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Matches question-number tokens at the start of a text string:
#   "3."   "3)"   "3:-"   "Q3"   "Q.3"   "Q3."
#   "Question 3:"   "Q.No. 3"   "ചോദ്യം 3"
_Q_HEADER_RE = re.compile(
    r"""^
    (?:
        (?:Question\s*(?:No\.?)?\s*|Ques\.?\s*(?:No\.?)?\s*|Q\.?\s*(?:No\.?)?\s*|Sl\.?\s*No\.?\s*|Problem\s*|Prob\.?\s*|ചോദ്യം\s*(?:നമ്പർ)?\s*)
        (\d{1,3})
        \s*
        (?:[.):\-]|--)?
        |
        (\d{1,3})
        \s*
        (?:[.):\-]|--)
    )
    (?!\d)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Section-header patterns: "Section A", "Part I", "Part B", "വിഭാഗം A"
_SECTION_RE = re.compile(
    r"""^(?:Section|Part|Group|Unit|Chapter|വിഭാഗം|ഖണ്ഡിക)\s*
         [-:]?\s*
         ([A-Za-z0-9IVX]+)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_bold(flags: int) -> bool:
    """PyMuPDF span flags: bit 4 (value 16) = bold."""
    return bool(flags & 16)


def _body_font_size(elements: list[dict]) -> float:
    """Estimate the dominant body font size via median of all text elements."""
    sizes = sorted(
        e["font_size"] for e in elements
        if e.get("type") == "text" and e.get("font_size", 0) > 0
    )
    if not sizes:
        return 11.0
    return sizes[len(sizes) // 2]


# ---------------------------------------------------------------------------
# Reading Order & Structural Math Detection
# ---------------------------------------------------------------------------

def sort_blocks_in_reading_order(blocks: list[dict]) -> list[dict]:
    """
    Sort blocks into true visual reading order by grouping horizontally
    overlapping spans into visual lines, sorting lines top-to-bottom,
    and sorting spans within each line left-to-right (by x0).
    """
    if not blocks:
        return []
    
    pages = {}
    for b in blocks:
        pages.setdefault(b.get("page", 0), []).append(b)
        
    sorted_all = []
    for p in sorted(pages.keys()):
        p_blocks = pages[p]
        p_blocks = sorted(p_blocks, key=lambda b: (b.get("bbox", [0, 0, 0, 0])[1], b.get("bbox", [0, 0, 0, 0])[0]))
        
        lines = []
        for b in p_blocks:
            bbox = b.get("bbox", [0, 0, 0, 0])
            b_y0, b_y1 = bbox[1], bbox[3]
            b_yc = (b_y0 + b_y1) / 2.0
            b_h = max(1.0, b_y1 - b_y0)
            
            matched_line = None
            for line in lines:
                line_y0 = min(x.get("bbox", [0, 0, 0, 0])[1] for x in line)
                line_y1 = max(x.get("bbox", [0, 0, 0, 0])[3] for x in line)
                line_yc = (line_y0 + line_y1) / 2.0
                line_h = max(1.0, line_y1 - line_y0)
                
                # Check for significant horizontal collision with existing line elements.
                # If block b starts at the same horizontal position as an element in line, they are vertically stacked separate lines!
                has_horizontal_collision = any(
                    max(bbox[0], x.get("bbox", [0, 0, 0, 0])[0]) <
                    min(bbox[2], x.get("bbox", [0, 0, 0, 0])[2]) - 3.0
                    for x in line
                )
                if has_horizontal_collision:
                    continue

                overlap = min(b_y1, line_y1) - max(b_y0, line_y0)
                if overlap > 0.4 * min(b_h, line_h) or abs(b_yc - line_yc) < min(b_h, line_h) * 0.45:
                    matched_line = line
                    break
            if matched_line is not None:
                matched_line.append(b)
            else:
                lines.append([b])
                
        lines = sorted(lines, key=lambda l: (min(x.get("bbox", [0, 0, 0, 0])[1] for x in l), min(x.get("bbox", [0, 0, 0, 0])[0] for x in l)))
        for line in lines:
            line_sorted = sorted(line, key=lambda x: x.get("bbox", [0, 0, 0, 0])[0])
            sorted_all.extend(line_sorted)
            
    return sorted_all


def _flag_structural_regions(span_blocks: list[dict]) -> list[dict]:
    """
    Detect fractions and subscripts geometrically and flag them for structural OCR.
    All thresholds are dynamically scaled according to local font size and geometry.
    """
    processed = []
    used_ids = set()
    
    # Simple ID assignment
    for i, sb in enumerate(span_blocks):
        sb["_id"] = i
        
    lines = [sb for sb in span_blocks if sb.get("type") == "line"]
    texts = [sb for sb in span_blocks if sb.get("type") == "text"]
    body_size = _body_font_size(span_blocks)
    
    # 0. Detect Math/Symbol fonts and Garbled Text
    for t in texts:
        font_name = str(t.get("font", "")).lower()
        t_text = str(t.get("text", ""))
        
        is_math_font = any(f in font_name for f in ["math", "symbol", "cmsy", "cmmi", "cmex", "cmr", "euclid", "std", "symb", "greek", "mt", "asana", "lucida", "fira"])
        
        # Heuristic: PyMuPDF often mis-maps CID math fonts to Malayalam Unicode block (U+0D00 - U+0D7F).
        # If we see Malayalam digits (0D66-0D6F) or if Malayalam chars are mixed heavily with Latin/math, it's garbled math.
        has_malayalam = bool(re.search(r'[\u0D00-\u0D7F]', t_text))
        has_malayalam_digits = bool(re.search(r'[\u0D66-\u0D6F]', t_text))
        has_math_operators = bool(re.search(r'[\=\+\-\*\/\^\_\±\×\÷\√]', t_text))
        
        is_garbled_math = has_malayalam_digits or (has_malayalam and (has_math_operators or bool(re.search(r'[a-zA-Z]{1,2}', t_text))))
        
        if is_math_font or is_garbled_math:
            t["needs_structural_ocr"] = True
            t["region_type"] = "inline_math"

    # 1. Universal Fraction Detection
    for line in lines:
        if line["_id"] in used_ids: continue
        lx0, ly0, lx1, ly1 = line["bbox"]
        ly_center = (ly0 + ly1) / 2
        l_w = lx1 - lx0
        if l_w <= 0 or l_w > 15 * body_size: continue
        
        above = []
        below = []
        for t in texts:
            if t["_id"] in used_ids: continue
            if t.get("page") != line.get("page"): continue
            tx0, ty0, tx1, ty1 = t.get("bbox", [0, 0, 0, 0])
            t_sz = t.get("font_size", body_size)
            
            # Text span must align horizontally with the fraction bar
            if tx0 >= lx0 - 0.4 * t_sz and tx1 <= lx1 + 0.4 * t_sz:
                # Text above fraction bar
                if ly_center - 1.3 * t_sz <= ty0 and ty1 <= ly_center + 0.15 * t_sz:
                    above.append(t)
                # Text below fraction bar
                elif ly_center - 0.15 * t_sz <= ty0 and ty1 <= ly_center + 1.5 * t_sz:
                    below.append(t)
                    
        if above and below:
            # Fraction components must not contain option markers or marks brackets
            if any(re.search(r'\([a-dA-D1-4]\)|\[\s*\d+\s*marks?\s*\]', p.get("text", ""), re.IGNORECASE) for p in above + below):
                continue
            all_parts = [line] + above + below
            for p in all_parts:
                used_ids.add(p["_id"])
            
            min_x = min(p["bbox"][0] for p in all_parts)
            min_y = min(p["bbox"][1] for p in all_parts)
            max_x = max(p["bbox"][2] for p in all_parts)
            max_y = max(p["bbox"][3] for p in all_parts)
            
            above_text = "".join(p.get("text", "") for p in sorted(above, key=lambda x: x["bbox"][0]))
            below_text = "".join(p.get("text", "") for p in sorted(below, key=lambda x: x["bbox"][0]))
            
            processed.append({
                "type": "equation",
                "text": f"\\frac{{{above_text}}}{{{below_text}}}",
                "bbox": [min_x, min_y, max_x, max_y],
                "page": line["page"],
                "font_size": above[0].get("font_size", body_size),
                "is_bold": False,
                "x_indent": min_x / max(1, min_x + 100),
                "needs_structural_ocr": True,
                "region_type": "fraction"
            })
            
    # 2. Universal Subscript / Superscript Detection
    for t in texts:
        if t["_id"] in used_ids:
            continue
        tx0, ty0, tx1, ty1 = t.get("bbox", [0, 0, 0, 0])
        ty_center = (ty0 + ty1) / 2
        t_sz = t.get("font_size", body_size)
        
        neighbors = []
        for other in texts:
            if other["_id"] in used_ids or other["_id"] == t["_id"]: continue
            if other.get("page") != t.get("page"): continue
            
            ox0, oy0, ox1, oy1 = other.get("bbox", [0, 0, 0, 0])
            oy_center = (oy0 + oy1) / 2
            o_sz = other.get("font_size", body_size)
            
            # Horizontally adjacent within font-proportional distance:
            # Sub/sup must be strictly to the RIGHT of base text
            if tx1 - 0.2 * t_sz <= ox0 <= tx1 + 0.8 * t_sz:
                # Smaller font size and vertically offset baseline
                if o_sz < t_sz * 0.95 and 0.12 * t_sz <= abs(oy_center - ty_center) <= 0.65 * t_sz:
                    # Must overlap vertically with the base text line (not line above or line below)
                    if oy0 < ty1 and oy1 > ty0:
                        # Do not absorb option markers or brackets
                        if not re.search(r'^\([a-dA-D1-4]\)$', other.get("text", "").strip()):
                            neighbors.append(other)
                    
        if neighbors:
            all_parts = [t] + neighbors
            for p in all_parts:
                used_ids.add(p["_id"])
                
            min_x = min(p["bbox"][0] for p in all_parts)
            min_y = min(p["bbox"][1] for p in all_parts)
            max_x = max(p["bbox"][2] for p in all_parts)
            max_y = max(p["bbox"][3] for p in all_parts)
            
            combined_text = t.get("text", "")
            for other in sorted(neighbors, key=lambda x: x["bbox"][0]):
                o_yc = (other["bbox"][1] + other["bbox"][3]) / 2
                o_txt = other.get("text", "").strip()
                if o_yc < ty_center:
                    combined_text += f"$^{{{o_txt}}}$"
                else:
                    combined_text += f"$_{{{o_txt}}}$"
                    
            processed.append({
                "type": "text",
                "text": combined_text,
                "bbox": [min_x, max(min_y, ty0 - 0.3 * t_sz), max_x, min(max_y, ty1 + 0.3 * t_sz)],
                "page": t["page"],
                "font_size": t_sz,
                "is_bold": t.get("is_bold", False),
                "x_indent": t.get("x_indent", 0.0),
                "needs_structural_ocr": False,
                "region_type": "sub_sup"
            })
            
    # Add remaining blocks
    for sb in span_blocks:
        if sb.get("_id") not in used_ids and sb.get("type") != "line":
            processed.append(sb)
            
    # Re-sort blocks in proper reading order
    return sort_blocks_in_reading_order(processed)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def segment_questions(span_blocks: list[dict]) -> list[QuestionSegment]:
    """
    Split a list of LayoutElement dicts (from extract_text_from_pdf's
    ``span_blocks`` return key) into QuestionSegment objects.

    Parameters
    ----------
    span_blocks:
        List of dicts with keys: type, text, bbox, page, font_size,
        is_bold (bool), x_indent (float 0-1), flags (int, optional).

    Returns
    -------
    list[QuestionSegment]
        One segment per detected question, in document order.
        Returns [] if no boundaries can be detected (caller should fall back
        to parse_questions_from_text on the flat text).
    """
    if not span_blocks:
        return []

    # 1. Structural Math Detection Pre-pass
    span_blocks = _flag_structural_regions(span_blocks)

    body_size = _body_font_size(span_blocks)

    segments: list[QuestionSegment] = []
    current_segment: Optional[QuestionSegment] = None
    current_section: Optional[str] = None
    prev_elem: Optional[LayoutElement] = None

    for raw in span_blocks:
        text = (raw.get("text") or "").strip()
        if not text and not raw.get("needs_structural_ocr"):
            continue

        elem = LayoutElement(
            type=raw.get("type", "text"),
            text=text,
            bbox=raw.get("bbox", [0, 0, 0, 0]),
            page=raw.get("page", 0),
            font_size=raw.get("font_size", body_size),
            is_bold=raw.get("is_bold", False) or _is_bold(raw.get("flags", 0)),
            x_indent=raw.get("x_indent", 0.0),
            needs_structural_ocr=raw.get("needs_structural_ocr", False),
            region_type=raw.get("region_type", "")
        )

        # -----------------------------------------------------------------
        # Detect section headers — capture but do not start a new segment
        # -----------------------------------------------------------------
        sec_m = _SECTION_RE.match(text)
        if sec_m and elem.font_size >= body_size * 1.05:
            current_section = text
            if current_segment is not None:
                current_segment.elements.append(elem)
            continue

        ans_m = re.search(r'(?i)^(?:answer\s*key|answers|solutions|ഉത്തരങ്ങൾ|പരിഹാരങ്ങൾ|hints)\s*$', text)
        if ans_m and elem.font_size >= body_size * 1.05:
            current_section = "Answer Key"
            if current_segment is not None:
                current_segment.elements.append(elem)
            continue

        # -----------------------------------------------------------------
        # Test whether this element is a question-boundary marker
        # -----------------------------------------------------------------
        q_m = _Q_HEADER_RE.match(text)
        if q_m:
            q_num = int(q_m.group(1) or q_m.group(2))
            remainder = text[q_m.end():]
            gap_size = len(remainder) - len(remainder.lstrip())
            
            if 1 <= q_num <= 300 and _is_question_boundary(elem, body_size, gap_size, prev_elem, current_segment):
                # Start a new segment
                if current_segment is not None and current_segment.elements:
                    segments.append(current_segment)
                current_segment = QuestionSegment(
                    q_num=q_num,
                    page=elem.page,
                    section=current_section,
                    elements=[elem],
                )
                prev_elem = elem
                continue

        # -----------------------------------------------------------------
        # Not a boundary — accumulate into the current segment
        # -----------------------------------------------------------------
        if current_segment is not None:
            current_segment.elements.append(elem)
        # Elements before the first question boundary (exam headers/instructions)
        # are silently dropped here.
        
        prev_elem = elem

    # Flush the last open segment
    if current_segment is not None and current_segment.elements:
        segments.append(current_segment)

    # Post-filter: remove implausible sequence members
    segments = _filter_valid_segments(segments)

    return segments


# ---------------------------------------------------------------------------
# Structural corroboration test
# ---------------------------------------------------------------------------

def _is_question_boundary(elem: LayoutElement, body_size: float, gap_size: int, prev_elem: Optional[LayoutElement], current_segment: Optional[QuestionSegment]) -> bool:
    """
    Return True only if structural cues support this element being a
    Return True only if structural cues support this element being a
    question-header, not just a numbered list item inside explanation text.
    """
    if elem.is_bold:
        return True

    # Check if this line is indented noticeably further than the current question's start.
    # If it is, it's a wrapped line inside a paragraph (e.g., "  3. Loop Q") rather than a new question.
    base_indent = current_segment.elements[0].x_indent if current_segment and current_segment.elements else 0.0
    is_indented_wrap = (current_segment is not None) and (elem.x_indent > base_indent + 0.02)

    # We do NOT return yet. We must first evaluate if it's a continuation of the previous line.

    # Check if the current segment already contains an 'Answer:' or 'Ans:' block,
    # or if prev_elem is an answer line.
    has_answer = False
    if current_segment:
        for e in current_segment.elements:
            has_keyword = bool(re.search(r'^(?:Answer|Ans|ഉത്തരം|Solution|Explanation)(?:[\s:\.\-]*)$', e.text.strip(), re.IGNORECASE))
            has_prefix = bool(re.search(r'^(?:Answer|Ans|ഉത്തരം|Solution|Explanation)[\s:\.\-]+', e.text.strip(), re.IGNORECASE))
            if has_keyword or has_prefix or (e.is_bold and e.text.strip().lower() in ('answer', 'ans', 'solution', 'explanation', 'ഉത്തരം')):
                has_answer = True
                break

    if prev_elem and re.search(r'^(?:Answer|Ans|ഉത്തരം|Solution|Explanation)[\s:\.\-]+', prev_elem.text.strip(), re.IGNORECASE):
        has_answer = True

    # If the current question already has an answer, the next numbered item is likely the start of a new question!
    # We shouldn't penalize it for being indented, but we still subject it to basic continuity checks.
    if has_answer:
        is_indented_wrap = False

    # Vertical Distance Check: If there is a noticeable vertical gap (e.g. 2 lines of space)
    # between the previous block and this block, it signifies a paragraph break.
    y_gap = 0.0
    is_new_page = False
    is_continuation = False
    if prev_elem:
        if elem.page != prev_elem.page:
            is_new_page = True
        else:
            y_gap = elem.bbox[1] - prev_elem.bbox[3]
            
        # If the previous text ends with punctuation or words strongly implying a wrapped sentence,
        # we track it to prevent false positive splits on typos.
        txt = prev_elem.text.strip()
        if txt:
            if txt.endswith((':', ',', '-', '∶', '::')):
                is_continuation = True
            else:
                last_word = txt.split()[-1].lower()
                if last_word in ('the', 'a', 'an', 'and', 'or', 'of', 'in', 'to', 'for', 'with', 'is', 'are', 'ratio'):
                    is_continuation = True
                
    # If this is physically consecutive text AND it follows a continuation marker, it is a wrapped line.
    if is_continuation and y_gap <= (body_size * 0.8) and not is_new_page:
        return False

    if not is_indented_wrap:
        if gap_size >= 1 or gap_size == 0 or elem.x_indent < 0.25 or (0.45 < elem.x_indent < 0.60):
            return True
            
        if is_new_page or y_gap > (body_size * 0.8):
            return True

    return False


# ---------------------------------------------------------------------------
# Sequence sanity filter
# ---------------------------------------------------------------------------

def _filter_valid_segments(segments: list[QuestionSegment]) -> list[QuestionSegment]:
    """
    Remove segments whose q_num is implausible given their neighbours.
    Mirrors the logic in find_sequential_question_boundaries() but operates
    on QuestionSegment objects.

    Rules:
    - Accept q_num == 1 (or the first seen number) as the anchor.
    - Accept forward jumps up to +25 (handles missing questions).
    - Accept deliberate section restarts (q_num drops back after a section
      header change).
    - Discard anything else (likely a false positive inside an explanation).
    """
    if not segments:
        return []

    # Find a good starting anchor — prefer q_num == 1 in the first 6
    start_idx = 0
    for i, seg in enumerate(segments[:6]):
        if seg.q_num == 1:
            start_idx = i
            break

    accepted = [segments[start_idx]]
    expected = segments[start_idx].q_num + 1
    prev_section = segments[start_idx].section

    for i, seg in enumerate(segments[start_idx + 1:], start=start_idx + 1):
        q = seg.q_num
        section_changed = seg.section != prev_section and seg.section is not None

        is_valid = False
        if q == expected:
            is_valid = True
            expected = q + 1
        elif section_changed and q <= expected:
            # Section restart (e.g. "Part B" resets counter to 1)
            is_valid = True
            expected = q + 1
        elif q == 1 and not section_changed:
            # Possible exam restart without a section header (e.g. after instructions)
            # Accept if it's on a new page, OR if it initiates a long sequence (1,2,3,4)
            is_new_page = accepted and seg.page > accepted[-1].page
            future_qs = [s.q_num for s in segments[i + 1 : i + 12]]
            has_long_sequence = 2 in future_qs and 3 in future_qs and 4 in future_qs
            if is_new_page or has_long_sequence:
                is_valid = True
                expected = 2
        elif q > expected:
            # Check lookahead: if a future segment (within next 10) matches 'expected' or 'expected+1' or 'expected+2',
            # then THIS 'q' is a false positive jump (e.g. we see 90, but 6 is coming up).
            future_qs = [s.q_num for s in segments[i + 1 : i + 12]]
            if any(fq in (expected, expected + 1, expected + 2) for fq in future_qs):
                is_valid = False
            else:
                is_valid = True
                expected = q + 1
        # Else: q < expected and not section_changed -> backward jump (false positive) -> is_valid = False

        if is_valid:
            accepted.append(seg)
        else:
            # DISCARDED as a new question, BUT we must not lose the text!
            # It's a false positive boundary (e.g. a numbered list inside the previous question).
            # Merge its elements into the last accepted segment.
            if accepted:
                accepted[-1].elements.extend(seg.elements)

        prev_section = seg.section

    return accepted
