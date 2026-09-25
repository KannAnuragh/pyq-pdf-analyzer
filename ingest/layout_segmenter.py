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
#   "Question 3:"   "Q.No. 3"
_Q_HEADER_RE = re.compile(
    r"""^
    (?:
        (?:Question\s*(?:No\.?)?\s*|Q\.?\s*(?:No\.?)?\s*|Sl\.?\s*No\.?\s*|Problem\s*|Prob\.?\s*)?
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

    body_size = _body_font_size(span_blocks)

    segments: list[QuestionSegment] = []
    current_segment: Optional[QuestionSegment] = None
    current_section: Optional[str] = None
    prev_elem: Optional[LayoutElement] = None

    for raw in span_blocks:
        text = (raw.get("text") or "").strip()
        if not text:
            continue

        elem = LayoutElement(
            type=raw.get("type", "text"),
            text=text,
            bbox=raw.get("bbox", [0, 0, 0, 0]),
            page=raw.get("page", 0),
            font_size=raw.get("font_size", body_size),
            is_bold=raw.get("is_bold", False) or _is_bold(raw.get("flags", 0)),
            x_indent=raw.get("x_indent", 0.0),
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

        # -----------------------------------------------------------------
        # Test whether this element is a question-boundary marker
        # -----------------------------------------------------------------
        q_m = _Q_HEADER_RE.match(text)
        if q_m:
            q_num = int(q_m.group(1))
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

    # Check if the current segment already contains an 'Answer:' or 'Ans:' block.
    # If it does, the next numbered item is extremely likely to be the start of a new question,
    # regardless of horizontal gap sizes.
    has_answer = False
    if current_segment:
        for e in current_segment.elements:
            if re.search(r'^(?:Answer|Ans|ഉത്തരം)[\s:]', e.text, re.IGNORECASE):
                has_answer = True
                break

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
            last_word = txt.split()[-1].lower()
            if last_word.endswith((':', ',', '-', 'the', 'a', 'an', 'and', 'or', 'of', 'in', 'to', 'for', 'with', 'is', 'are', 'ratio', '∶', '::')):
                is_continuation = True
                
    # If this is physically consecutive text AND it follows a continuation marker, it is a wrapped line.
    if is_continuation and y_gap <= (body_size * 0.8) and not is_new_page:
        return False

    if not is_indented_wrap:
        if gap_size >= 2 or gap_size == 0 or elem.x_indent < 0.20:
            return True
            
        if is_new_page or (has_answer and y_gap > body_size * 0.5) or y_gap > (body_size * 0.8):
            return True

    return False
    # a 1-space gap (gap_size == 1) is almost certainly just a wrapped line inside the paragraph.
    # Therefore, we NEVER treat it as a question boundary without a distinct paragraph break.
    
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

    for seg in segments[start_idx + 1:]:
        q = seg.q_num
        section_changed = seg.section != prev_section and seg.section is not None

        if q == expected:
            accepted.append(seg)
            expected = q + 1
        elif q > expected and q <= expected + 25:
            # Forward gap — acceptable (some questions may be omitted)
            accepted.append(seg)
            expected = q + 1
        elif section_changed and q <= expected:
            # Section restart (e.g. "Part B" resets counter to 1)
            accepted.append(seg)
            expected = q + 1
        # Else: discard (likely a number inside an explanation or table)

        prev_section = seg.section

    return accepted
