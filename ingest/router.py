"""
Universal PDF text extraction router.

Extraction strategy per page:
1. Detect inline math/equation images — validate and preserve as crisp base64 PNGs
2. Detect ALL subset fonts → build/retrieve CID→Unicode maps (font_mapper)
3. Rebuild text using font maps (rawdict char-by-char), tracking unmapped codes
4. Clean text (normalize, visual-to-logical reorder)
5. Return per-page metadata (text, font_hashes, unmapped_codes, spans) for
   downstream per-question quality scoring and tiered repair

NOTE: Page-level sanity check + full-page Vision OCR fallback has been REMOVED.
Quality scoring and repair are now handled per-question in the graph pipeline
(quality_score_node → repair_node) which diagnoses root causes and applies
the cheapest viable fix before resorting to Vision calls.
"""

import pymupdf
import unicodedata
import os
import pytesseract
import base64
import hashlib
import re
from typing import Optional, List, Dict, Tuple
from .text_cleaner import convert_visual_to_logical, clean_indic_text, sanity_check_malayalam_text
from .quality_check import validate_equation_crop
from ..storage.cloudflare import upload_image_to_cloudflare
from google import genai
from google.genai import types
from PIL import Image
import io
try:
    from groq import Groq
except ImportError:
    Groq = None


_quota_exhausted = False
_groq_vision_supported = True


def clean_fraction_term(term: str) -> str:
    """Cleans OCR noise from numerator/denominator terms."""
    term = term.strip()
    term = re.sub(r'^[|!_~—–\.\s]+', '', term)
    term = re.sub(r'[|!_~—–\.\s]+$', '', term)
    term = term.replace(" ", "")
    return term if term else "1"


KNOWN_LATEX_COMMANDS = {
    'frac', 'dfrac', 'tfrac', 'sqrt', 'times', 'div', 'pm', 'mp', 'cdot', 'circ', 'degree',
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa',
    'lambda', 'mu', 'nu', 'xi', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi',
    'omega', 'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon', 'Phi', 'Psi',
    'Omega', 'neq', 'ne', 'approx', 'leq', 'geq', 'le', 'ge', 'infty', 'sum', 'prod', 'int',
    'log', 'ln', 'sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'arcsin', 'arccos', 'arctan',
    'sinh', 'cosh', 'tanh', 'in', 'notin', 'partial', 'nabla', 'vec', 'hat', 'lim', 'det',
    'begin', 'end', 'left', 'right', 'bmatrix', 'pmatrix', 'vmatrix', 'Vmatrix', 'Bmatrix', 'matrix', 'array',
    'sim', 'bar', 'tilde', 'hbar', 'rightarrow', 'to', 'leftrightharpoons', 'rightleftharpoons', 'leftrightarrow',
    'text', 'mathrm', 'mathbf', 'mathbb', 'mathit', 'quad', 'qquad', 'equiv', 'propto', 'perp', 'parallel',
    'cup', 'cap', 'subset', 'subseteq', 'supset', 'supseteq', 'emptyset', 'forall', 'exists', 'implies', 'iff'
}


# ---------------------------------------------------------------------------
# Anchor token helpers
# ---------------------------------------------------------------------------

# Images and equations are stored as canonical anchor tokens in the span
# stream (and in the DB).  The API layer calls expand_anchor_tokens() at
# serialisation time so the frontend always receives ready-to-render HTML.
#
# Token formats:
#   {{IMG:https://cdn.example.com/abc.png}}   — Cloudflare R2 image
#   {{EQ:$\frac{1}{2}$}}                      — inline LaTeX equation

_ANCHOR_IMG_RE = re.compile(r'\{\{\s*IMG\s*:\s*((?:https?://|data:image/)[^\s}]+)\}*')
_ANCHOR_EQ_RE  = re.compile(r'\{\{EQ:([^}]+)\}*\}')


def expand_anchor_tokens(text: str) -> str:
    """
    Convert anchor tokens to HTML for frontend consumption.

    {{IMG:url}}         -> <img src="url" class="inline-math-img" ...>
    {{EQ:$latex$}}      -> $latex$   (kept as-is; KaTeX renders it)
    {{IMG_BLOCK:url}}   -> <div class="..."><img ...></div>
    """
    if not text:
        return text

    # Block diagram token (large images: supports CDN URLs and fallback data: URIs)
    text = re.sub(
        r'\{\{\s*IMG_BLOCK\s*:\s*((?:https?://|data:image/)[^\s}]+)\}*',
        r'![diagram](\1)',
        text,
    )

    # Inline image token
    text = _ANCHOR_IMG_RE.sub(
        r'![math](\1)',
        text,
    )

    # Equation token — already valid LaTeX, just unwrap the outer token wrapper
    text = _ANCHOR_EQ_RE.sub(lambda m: m.group(1), text)

    # Strip any un-restored QAI_IMAGE tokens so raw placeholders never show to users
    text = re.sub(r'\[\[\s*QAI_IMAGE:[^\]]+\]\]', '', text)

    # Strip any hallucinated example.com / placeholder tokens
    text = re.sub(r'\{\{\s*(?:IMG|IMG_BLOCK)\s*:\s*https?://(?:www\.)?example\.com[^\s}]*\}\}', '', text)
    text = re.sub(r'!\[(?:diagram|math)\]\(https?://(?:www\.)?example\.com[^\s)]*\)', '', text)

    return text


def _make_img_anchor(img_url: str, h: float, w: float, is_block: bool,
                     page_idx: int, owner_number=None) -> str:
    """
    Build the canonical anchor token for an image.  Block images use
    {{IMG_BLOCK:url}} so expand_anchor_tokens() can wrap them in a <div>.
    """
    if is_block:
        return f'{{{{IMG_BLOCK:{img_url}}}}}'
    return f'{{{{IMG:{img_url}}}}}'


def clean_latex_equation(raw_text: str) -> str:
    """Standardizes and sanitizes extracted math text into crisp, valid inline LaTeX ($...)."""
    if not raw_text:
        return ""
    cleaned = raw_text.strip()
    cleaned = re.sub(r'```(?:latex|tex|math)?', '', cleaned)
    cleaned = cleaned.replace('```', '').strip()

    # Extract inner math string from delimiters
    m_disp = re.search(r'\$\$([\s\S]+?)\$\$', cleaned)
    if m_disp:
        inner = m_disp.group(1).strip()
    else:
        m_bracket = re.search(r'\\\[([\s\S]+?)\\\]', cleaned)
        if m_bracket:
            inner = m_bracket.group(1).strip()
        else:
            m_inline = re.search(r'\$([^$]+)\$', cleaned)
            if m_inline:
                inner = m_inline.group(1).strip()
            else:
                m_paren = re.search(r'\\\(([\s\S]+?)\\\)', cleaned)
                if m_paren:
                    inner = m_paren.group(1).strip()
                else:
                    inner = cleaned.strip("$").strip()

    if not inner:
        return ""

    # Convert slash fraction like 1/2 to \frac{1}{2}
    slash_frac = re.match(r'^(\d+)\s*/\s*(\d+)$', inner)
    if slash_frac:
        inner = f"\\frac{{{slash_frac.group(1)}}}{{{slash_frac.group(2)}}}"

    # 1. Vector arrow in fraction numerator misrecognition
    # e.g. \frac{=>}{F}, \frac{>}{F}, \frac{ - }{F}, \frac{=d}{F}, \frac{=}{F}
    inner = re.sub(r'\\frac\{[=><\-—~^d2\s|!_]+\}\{([A-Za-z])(?:\s*[A-Za-z]*)\}', r'\\vec{\1}', inner)

    # 2. Radical / Square root misrecognitions from OCR (e.g. \X, VX, Vx -> \sqrt{x})
    inner = re.sub(r'\\X\b', r'\\sqrt{x}', inner)
    inner = re.sub(r'\bVX\b', r'\\sqrt{x}', inner)
    inner = re.sub(r'\bVx\b', r'\\sqrt{x}', inner)
    inner = re.sub(r'\bV(\d+)\b', r'\\sqrt{\1}', inner)
    inner = re.sub(r'\\x\b', 'x', inner)

    # 3. Multiplication & variable normalization
    inner = re.sub(r'×', r'\\times ', inner)
    inner = re.sub(r'(\d+(?:\^\d+)?)\s*[xX]\^([0-9])\b', r'\1 \\times \2', inner)
    inner = re.sub(r'(\d+(?:\.\d+)?|\^\{?\d+\}?)\s*[xX]\s*(\d+(?:\.\d+)?)', r'\1 \\times \2', inner)
    inner = re.sub(r'(\d+)\s+(?=\d+\s*=)', r'\1, ', inner)
    inner = re.sub(r'^\\times\s*=', 'x =', inner)
    # 4. Standardize spacing around equals
    inner = re.sub(r'\s*=\s*', ' = ', inner)

    # 5. Matrix healing: restore missing backslashes for begin/end/left/right and normalize array to bmatrix
    inner = re.sub(r"(?<!\\)\b(begin|end|left|right)\b", r"\\\1", inner)
    inner = re.sub(r"\\left\[\s*\\begin\{array\}\{[a-z]+\}\s*([\s\S]+?)\s*\\end\{array\}\s*\\right\]", r"\\begin{bmatrix} \1 \\end{bmatrix}", inner)

    # 6. Heal corrupted fractions (e.g. \frac{-q{4\pi -> \frac{-q}{4\pi})
    inner = re.sub(r'\\frac\{\\text\{([^{}]+)\}\{([^{}]+?)(?=\s*=\s*|\$|\\\\|\n|$)', r'\\frac{\\text{\1}}{\2}', inner)
    inner = re.sub(r'\\frac\{([^{}]+)\{([^{}]+?)(?=\s*=\s*|\$|\\\\|\n|$)', r'\\frac{\1}{\2}', inner)
    inner = re.sub(r'\\frac\s+([^{}\s]+)\{([^{}]+)\}', r'\\frac{\1}{\2}', inner)
    inner = re.sub(r'\\frac\s+([^{}\s]+)\{([^{}]+?)(?=\s*=\s*|\$|\\\\|\n|$)', r'\\frac{\1}{\2}', inner)

    # 7. Sanitize unknown control sequences that break KaTeX
    def _sanitize_cmd(match):
        cmd = match.group(1)
        if cmd in KNOWN_LATEX_COMMANDS:
            return match.group(0)
        if cmd.lower() in ('x', 'y', 'z', 'a', 'b', 'c', 'n'):
            return cmd.lower()
        return cmd

    from .math_normalizer import sanitize_math_expression
    inner = sanitize_math_expression(inner)
    result = f"${inner}$"

    # Structural sanity check on the way out: OCR'd math (ocr_math_image / Vision) is the
    # single biggest source of brace/delimiter corruption feeding into option text later
    # in the pipeline (e.g. "$\frac{\sqrt{2}}$kqq_0}{a^2}"). Catch it here, at the source,
    # rather than relying solely on the downstream option_answer_separator pass to notice.
    try:
        from .option_answer_separator import validate_and_repair_latex
        result, had_issue = validate_and_repair_latex(result)
        if had_issue:
            print(f"clean_latex_equation: repaired malformed LaTeX braces/delimiters in {result!r}", flush=True)
    except Exception:
        pass

    return result


def ocr_math_image(pil_img: Image.Image) -> str:
    """Local offline math OCR parser with fraction, exponent, variable, radical, and matrix detection."""
    try:
        import pytesseract
        from PIL import ImageOps
        gray = pil_img.convert("L")
        w, h = gray.size

        # 0. Matrix Detection: Images with brackets, aspect ratio < 2.3 and multi-row content
        if h >= 35 and (w / h) < 2.3:
            scaled = ImageOps.scale(gray, 3.0, resample=Image.Resampling.LANCZOS)
            th_mat = scaled.point(lambda p: 0 if p < 180 else 255)
            mat_txt = pytesseract.image_to_string(th_mat, config="--psm 11").strip()
            if ("[" in mat_txt or "]" in mat_txt):
                top_crop = gray.crop((int(w * 0.12), int(h * 0.05), int(w * 0.88), int(h * 0.52)))
                bot_crop = gray.crop((int(w * 0.12), int(h * 0.48), int(w * 0.88), int(h * 0.95)))
                t_sc = ImageOps.scale(top_crop, 3.0, resample=Image.Resampling.LANCZOS).point(lambda p: 0 if p < 180 else 255)
                b_sc = ImageOps.scale(bot_crop, 3.0, resample=Image.Resampling.LANCZOS).point(lambda p: 0 if p < 180 else 255)
                r1 = pytesseract.image_to_string(t_sc, config="--psm 7").strip().replace("[", "").replace("]", "").strip()
                r2 = pytesseract.image_to_string(b_sc, config="--psm 7").strip().replace("[", "").replace("]", "").strip()
                r1 = re.sub(r"\bS(?=\d)", "", r1)
                r2 = re.sub(r"\bS(?=\d)", "", r2)
                if r1 and r2 and any(c.isdigit() for c in r1) and any(c.isdigit() for c in r2):
                    row1_items = " & ".join(r1.split())
                    row2_items = " & ".join(r2.split())
                    return f"\\begin{{bmatrix}} {row1_items} \\\\ {row2_items} \\end{{bmatrix}}"

        # 0.5 Simple Operator Detection (Equals '=', Minus '-', Plus '+')
        if h <= 45 and w <= 100:
            row_sums = [sum(1 for x in range(w) if gray.getpixel((x, y)) < 160) for y in range(h)]
            max_row = max(row_sums) if row_sums else 0
            if max_row >= 6:
                bands = []
                in_band = False
                start_y = 0
                for y, count in enumerate(row_sums):
                    if count >= max_row * 0.4 and not in_band:
                        in_band = True
                        start_y = y
                    elif count < max_row * 0.4 and in_band:
                        in_band = False
                        bands.append((start_y, y - 1))
                if in_band:
                    bands.append((start_y, h - 1))

                if len(bands) == 2:
                    b1_w = sum(1 for x in range(w) if any(gray.getpixel((x, y)) < 160 for y in range(bands[0][0], bands[0][1] + 1)))
                    b2_w = sum(1 for x in range(w) if any(gray.getpixel((x, y)) < 160 for y in range(bands[1][0], bands[1][1] + 1)))
                    if b1_w >= 6 and b2_w >= 6:
                        return "="
                elif len(bands) == 1:
                    b_w = sum(1 for x in range(w) if any(gray.getpixel((x, y)) < 160 for y in range(bands[0][0], bands[0][1] + 1)))
                    col_sums_op = [sum(1 for y in range(h) if gray.getpixel((x, y)) < 160) for x in range(w)]
                    max_col_op = max(col_sums_op) if col_sums_op else 0
                    if b_w >= 8 and max_col_op <= h * 0.5:
                        return "-"
                    elif b_w >= 6 and max_col_op >= h * 0.65:
                        return "+"

        # 1. Whole image horizontal OCR (with pre-scaling for small clips)
        scaled_whole = ImageOps.scale(gray, 2.5, resample=Image.Resampling.LANCZOS) if (h < 70 or w < 160) else gray

        # Multi-line stacked equations (derivatives dy/dx, fractions 1/x, integrals, summations)
        if h >= 28:
            raw_multi = pytesseract.image_to_string(scaled_whole, config="--psm 6").strip()
            if raw_multi and (any(kw in raw_multi.lower() for kw in ["dy", "dx", "dt", "int", "sum", "lim", "/", "=", "+", "-"]) or "\n" in raw_multi):
                cleaned_multi = clean_latex_equation(raw_multi)
                if cleaned_multi and len(cleaned_multi) > 2 and not cleaned_multi.startswith("$=$"):
                    return cleaned_multi

        raw_whole = pytesseract.image_to_string(scaled_whole, config="--psm 7").strip()
        if not raw_whole or len(raw_whole) < 2:
            raw_whole = pytesseract.image_to_string(scaled_whole, config="--psm 6").strip()
        if not raw_whole or len(raw_whole) < 2:
            raw_whole = pytesseract.image_to_string(scaled_whole, config="--psm 11").strip()
        if raw_whole and any(c.isalnum() for c in raw_whole):
            cleaned = clean_latex_equation(raw_whole)
            if cleaned and len(cleaned) > 2 and not cleaned.startswith("$=$"):
                return cleaned

        # Check horizontal segments separated by whitespace
        col_sums = [sum(1 for y in range(h) if gray.getpixel((x, y)) < 128) for x in range(w)]

        segments = []
        start = None
        for x, c in enumerate(col_sums):
            if c > 0 and start is None:
                start = x
            elif c == 0 and start is not None:
                if x - start >= 3:
                    segments.append((start, x))
                start = None
        if start is not None and (w - start >= 3):
            segments.append((start, w))

        # Filter out speckles (width <= 3 and height <= 4)
        filtered_segments = []
        for x0, x1 in segments:
            crop = gray.crop((x0, 0, x1, h))
            cw, ch = crop.size
            active_y = [y for y in range(ch) if any(crop.getpixel((cx, y)) < 128 for cx in range(cw))]
            if active_y and (x1 - x0 > 3 or len(active_y) > 4):
                filtered_segments.append((x0, x1))
        segments = filtered_segments

        if len(segments) > 1:
            tokens = []
            prev_y_top = None
            prev_y_bot = None
            prev_token = None

            for idx, (x0, x1) in enumerate(segments):
                crop = gray.crop((max(0, x0 - 2), 0, min(w, x1 + 2), h))
                cw, ch = crop.size
                row_sums = [sum(1 for cx in range(cw) if crop.getpixel((cx, y)) < 128) for y in range(ch)]
                active_rows = [y for y, r in enumerate(row_sums) if r > 0]

                if not active_rows:
                    continue

                y_top = active_rows[0]
                y_bot = active_rows[-1]
                y_span = y_bot - y_top

                # 1. Equals & Operator detection (=, ~, ,)
                crop_10 = pytesseract.image_to_string(crop, config="--psm 10").strip()
                crop_6 = pytesseract.image_to_string(crop, config="--psm 6").strip()
                t_raw = crop_10 or crop_6

                if "~" in t_raw:
                    tokens.append("\\sim")
                    prev_y_top, prev_y_bot, prev_token = y_top, y_bot, "\\sim"
                    continue
                if "," in t_raw:
                    tokens.append(",")
                    prev_y_top, prev_y_bot, prev_token = y_top, y_bot, ","
                    continue
                if "=" in t_raw:
                    tokens.append("=")
                    prev_y_top, prev_y_bot, prev_token = y_top, y_bot, "="
                    continue

                # 2. Fraction or Vector check inside segment
                lines = [l.strip() for l in crop_6.split("\n") if l.strip()]
                if len(lines) == 3 and lines[1] in ("-", "—", "–", "_", "="):
                    n = clean_fraction_term(lines[0])
                    d = clean_fraction_term(lines[2])
                    tok = f"\\frac{{{n}}}{{{d}}}"
                    tokens.append(tok)
                    prev_y_top, prev_y_bot, prev_token = y_top, y_bot, tok
                    continue
                elif len(lines) == 2:
                    # Check if line 0 is a vector arrow e.g. '—\nF' or '=>\nF' or '=d\nF'
                    if any(arr in lines[0] for arr in ['>', '=>', '->', '=d', '—', '=2', '^', '~', '-']) and re.match(r'^[A-Za-z]', lines[1]):
                        m_vec = re.match(r'^([A-Za-z])(?:\s*=\s*(.*))?$', lines[1])
                        if m_vec:
                            vec_var = m_vec.group(1).upper()
                            rest = m_vec.group(2)
                            tok = f"\\vec{{{vec_var}}} = {rest}" if rest else f"\\vec{{{vec_var}}}"
                            tokens.append(tok)
                            prev_y_top, prev_y_bot, prev_token = y_top, y_bot, tok
                            continue
                    n = clean_fraction_term(lines[0])
                    d = clean_fraction_term(lines[1])
                    tok = f"\\frac{{{n}}}{{{d}}}"
                    tokens.append(tok)
                    prev_y_top, prev_y_bot, prev_token = y_top, y_bot, tok
                    continue

                # 3. Single token classification
                t = t_raw.strip()

                # Check for radical sign (VX, Vx, \X, \x, 1X, 1x)
                m_rad = re.match(r'^(?:[Vv\\\/]|1)([a-zA-Z]|\d+)$', t)
                if m_rad:
                    var = m_rad.group(1).lower()
                    tok = f"\\sqrt{{{var}}}"
                    tokens.append(tok)
                    prev_y_top, prev_y_bot, prev_token = y_top, y_bot, tok
                    continue

                # Exponent / superscript check:
                # If segment's bottom is in upper half of line (y_bot <= ch * 0.48)
                # and previous token was a variable or number
                is_superscript = (
                    prev_token is not None and
                    prev_token not in ("=", "+", "-", "\\times", "\\div", "\\pm", "(", ")") and
                    (y_bot <= ch * 0.48 or (prev_y_bot is not None and y_bot < prev_y_bot - ch * 0.25))
                )

                if t in ("-", "—", "–"):
                    tok = "-"
                elif t in ("*", "×"):
                    tok = "\\times"
                elif t in ("+", "±"):
                    tok = t
                elif t.lower() in ("x", "y", "z", "a", "b", "c", "n", "m", "t", "r"):
                    # Preserve variables cleanly - never convert to \times!
                    tok = t.lower()
                else:
                    tok = t

                # Unit vectors with hat after parenthesis or variable
                if tok.lower() in ("i", "j", "k") and prev_token in (")", "^{2}", "^2", "z", "x", "y"):
                    tok = f"\\hat{{{tok.lower()}}}"

                if is_superscript and tok and tok.isalnum():
                    tokens.append(f"^{{{tok}}}")
                else:
                    tokens.append(tok)

                prev_y_top = y_top
                prev_y_bot = y_bot
                prev_token = tok

            if tokens:
                res = " ".join(tokens)
                res = re.sub(r'\s*\^\{([^}]+)\}', r'^{\1}', res)
                res = re.sub(r'\s*=\s*', ' = ', res)
                res = re.sub(r'\s*([+-])\s*', r' \1 ', res)
                return clean_latex_equation(f"${res.strip()}$")

        # Single block fallback
        raw = pytesseract.image_to_string(gray, config="--psm 6").strip()
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        if len(lines) == 3 and lines[1] in ("-", "—", "–", "_", "="):
            n = clean_fraction_term(lines[0])
            d = clean_fraction_term(lines[2])
            return clean_latex_equation(f"$\\frac{{{n}}}{{{d}}}$")
        elif len(lines) == 2:
            # Check vector
            if any(arr in lines[0] for arr in ['>', '=>', '->', '=d', '—', '=2', '^', '~', '-']) and re.match(r'^[A-Za-z]', lines[1]):
                m_vec = re.match(r'^([A-Za-z])(?:\s*=\s*(.*))?$', lines[1])
                if m_vec:
                    vec_var = m_vec.group(1).upper()
                    rest = m_vec.group(2)
                    if rest:
                        return clean_latex_equation(f"$\\vec{{{vec_var}}} = {rest}$")
                    return clean_latex_equation(f"$\\vec{{{vec_var}}}$")
            n = clean_fraction_term(lines[0])
            d = clean_fraction_term(lines[1])
            if len(n) == 1 and n.isdigit() and len(d) == 1 and d.isalpha():
                return clean_latex_equation(f"${d.lower()}^{{{n}}}$")
            return clean_latex_equation(f"$\\frac{{{n}}}{{{d}}}$")
        elif len(lines) == 1:
            val = lines[0].strip()
            m_rad = re.match(r'^(?:y\s*=\s*)?(?:[Vv\\\/]|1)([a-zA-Z]|\d+)$', val)
            if m_rad:
                var = m_rad.group(1).lower()
                if "y" in val.lower() and "=" in val:
                    return clean_latex_equation(f"$y = \\sqrt{{{var}}}$")
                return clean_latex_equation(f"$\\sqrt{{{var}}}$")
            return clean_latex_equation(f"${val}$")
    except Exception as e:
        print(f"[Local Math OCR] Failed: {e}", flush=True)
    return ""


def ocr_equation_clip(img_bytes: bytes, engine: Optional[str] = None) -> tuple[str, int, int]:
    """
    Routes an equation / symbol image region based on the requested engine or runtime config:
    - 'tesseract': High-accuracy local Tesseract OCR ($0, offline fractions, roots, matrices, vectors)
    - 'groq': Groq Vision OCR (Qwen/Llama)
    - 'gemini': Gemini Vision OCR
    - 'mathpix': Mathpix API
    - 'off': Completely skips equation OCR
    - 'auto': Waterfall fallback (Tesseract -> Mathpix -> Groq -> Gemini)
    """
    global _quota_exhausted, _groq_vision_supported

    # Resolve active engine & enabled status
    if engine is None:
        try:
            from ..storage.config_manager import get_setting
            if not get_setting("math_ocr_enabled", True):
                return "", 0, 0
            engine = get_setting("math_ocr_engine", "auto")
        except Exception:
            engine = os.getenv("MATH_OCR_ENGINE", "auto")

    engine = str(engine or "auto").strip().lower()
    if engine in ("off", "disabled"):
        return "", 0, 0

    prompt = (
        "Transcribe the mathematical formula, equation, fraction, or symbol in this image snippet into standard inline LaTeX ($...$). "
        "Output ONLY the inline LaTeX enclosed in dollar signs (e.g. $\\frac{1}{3}$, $\\sqrt{x^2+y^2}$, $x^2$, or $\\chi^2$). "
        "No explanation, no extra text, no markdown backticks."
    )

    def _try_tesseract() -> tuple[str, int, int]:
        try:
            pil_img = Image.open(io.BytesIO(img_bytes))
            math = ocr_math_image(pil_img)
            if math:
                clean_math = clean_latex_equation(math)
                if clean_math:
                    return clean_math, 0, 0
        except Exception as e:
            print(f"[Equation OCR] Tesseract OCR failed: {e}", flush=True)
        return "", 0, 0

    def _try_mathpix() -> tuple[str, int, int]:
        app_id = os.getenv("MATHPIX_APP_ID")
        app_key = os.getenv("MATHPIX_APP_KEY")
        if app_id and app_key:
            try:
                import requests
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                response = requests.post(
                    "https://api.mathpix.com/v3/text",
                    json={"src": f"data:image/png;base64,{img_b64}", "formats": ["text"]},
                    headers={"app_id": app_id, "app_key": app_key, "Content-Type": "application/json"},
                    timeout=10
                )
                if response.status_code == 200:
                    result_text = response.json().get("text", "").strip()
                    math = clean_latex_equation(result_text)
                    if math:
                        return math, 0, 0
            except Exception as e:
                print(f"[Mathpix OCR] Failed: {e}", flush=True)
        return "", 0, 0

    def _try_groq() -> tuple[str, int, int]:
        global _groq_vision_supported
        enable_groq = os.getenv("ENABLE_GROQ", "false").lower() in ("true", "1", "yes")
        groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if enable_groq and groq_key and not groq_key.startswith("your_") and _groq_vision_supported and Groq is not None:
            try:
                client = Groq(api_key=groq_key, timeout=10.0)
                b64_img = base64.b64encode(img_bytes).decode("utf-8")
                model_name = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                            ]
                        }
                    ],
                    temperature=0.1,
                    max_tokens=60,
                    timeout=8.0
                )
                raw_text = resp.choices[0].message.content.strip()
                math = clean_latex_equation(raw_text)
                p_tok = resp.usage.prompt_tokens if resp.usage else 0
                c_tok = resp.usage.completion_tokens if resp.usage else 0
                if math:
                    return math, p_tok, c_tok
            except Exception as e:
                err_str = str(e)
                if "not support" in err_str.lower() or "invalid" in err_str.lower() or "400" in err_str or "429" in err_str or "rate_limit" in err_str.lower():
                    _groq_vision_supported = False
                    print(f"[Equation OCR] Groq Vision rate-limited/unsupported ({e}).", flush=True)
                else:
                    print(f"[Equation OCR] Groq Vision failed: {e}.", flush=True)
        return "", 0, 0

    def _try_gemini() -> tuple[str, int, int]:
        global _quota_exhausted
        enable_gemini = os.getenv("ENABLE_GEMINI", "false").lower() in ("true", "1", "yes")
        google_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
        if enable_gemini and google_key and not google_key.startswith("your_") and not _quota_exhausted:
            try:
                client = genai.Client(api_key=google_key)
                model_name = os.getenv("GEMINI_EXTRACT_MODEL", "gemini-3.1-flash")
                res = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                        prompt
                    ]
                )
                math = clean_latex_equation(res.text)
                p_tok = res.usage_metadata.prompt_token_count if res.usage_metadata else 0
                c_tok = res.usage_metadata.candidates_token_count if res.usage_metadata else 0
                if math:
                    return math, p_tok, c_tok
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    _quota_exhausted = True
                print(f"[Equation OCR] Gemini Vision failed: {e}.", flush=True)
        return "", 0, 0

    # Targeted execution
    if engine == "tesseract":
        return _try_tesseract()
    elif engine == "groq":
        return _try_groq()
    elif engine == "gemini":
        return _try_gemini()
    elif engine == "mathpix":
        return _try_mathpix()

    # Default: 'auto' waterfall
    res, p, c = _try_mathpix()
    if res:
        return res, p, c

    res, p, c = _try_groq()
    if res:
        return res, p, c

    res, p, c = _try_gemini()
    if res:
        return res, p, c

    res, p, c = _try_tesseract()
    if res:
        return res, p, c

    return "", 0, 0


def ocr_page_vision(img_bytes: bytes, engine: Optional[str] = None, force: bool = False) -> tuple[str, int, int]:
    """Fallback OCR for full pages (e.g. scanned image-only PDFs) using configured question OCR engine."""
    try:
        from ..storage.config_manager import get_setting
        if not force and not get_setting("question_ocr_enabled", True) and engine is None:
            return "", 0, 0
        if engine is None:
            engine = get_setting("auto_whole_pdf_ocr_engine", get_setting("question_ocr_engine", "auto"))
    except Exception:
        if engine is None:
            engine = os.getenv("AUTO_WHOLE_PDF_OCR_ENGINE", os.getenv("QUESTION_OCR_ENGINE", "auto"))

    engine = str(engine or "auto").strip().lower()
    if engine in ("off", "disabled"):
        return "", 0, 0

    if engine == "auto":
        # Waterfall auto resolution: Groq -> Gemini -> Tesseract
        groq_on = os.getenv("ENABLE_GROQ", "false").lower() in ("true", "1", "yes")
        groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
        gemini_on = os.getenv("ENABLE_GEMINI", "false").lower() in ("true", "1", "yes")
        gemini_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
        if groq_on and groq_key and not groq_key.startswith("your_"):
            engine = "groq"
        elif gemini_on and gemini_key and not gemini_key.startswith("your_"):
            engine = "gemini"
        else:
            engine = "tesseract"

    if engine == "tesseract":
        try:
            import pytesseract
            from PIL import Image
            pil_img = Image.open(io.BytesIO(img_bytes))
            try:
                avail = pytesseract.get_languages()
            except Exception:
                avail = []
            tess_lang = "mal+eng" if "mal" in avail else "eng"
            text = pytesseract.image_to_string(pil_img, lang=tess_lang, config="--psm 6").strip()
            return text, 0, 0
        except Exception as e:
            print(f"[Page OCR] Local Tesseract OCR failed: {e}", flush=True)
            return "", 0, 0

    prompt = (
        "Extract all question paper text and multiple-choice questions from this scanned page faithfully. "
        "Keep question numbers, Malayalam/English text, options (A, B, C, D), and answers verbatim. "
        "Transcribe mathematical formulas into standard inline LaTeX ($...$). "
        "DO NOT include any conversational text, pleasantries, or explanations. Output ONLY the extracted text and nothing else."
    )

    if engine == "groq":
        enable_groq = os.getenv("ENABLE_GROQ", "false").lower() in ("true", "1", "yes")
        groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if enable_groq and groq_key and not groq_key.startswith("your_") and Groq is not None:
            try:
                client = Groq(api_key=groq_key, timeout=25.0)
                b64_img = base64.b64encode(img_bytes).decode("utf-8")
                model_name = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                            ]
                        }
                    ],
                    temperature=0.1,
                    max_tokens=2048,
                    timeout=20.0
                )
                txt = resp.choices[0].message.content.strip()
                p_tok = resp.usage.prompt_tokens if resp.usage else 0
                c_tok = resp.usage.completion_tokens if resp.usage else 0
                return txt, p_tok, c_tok
            except Exception as e:
                print(f"[Page OCR] Groq Vision failed: {e}", flush=True)

    if engine == "gemini":
        enable_gemini = os.getenv("ENABLE_GEMINI", "false").lower() in ("true", "1", "yes")
        google_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
        if enable_gemini and google_key and not google_key.startswith("your_"):
            try:
                client = genai.Client(api_key=google_key)
                model_name = os.getenv("GEMINI_EXTRACT_MODEL", "gemini-3.1-flash")
                res = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                        prompt
                    ]
                )
                txt = res.text.strip()
                p_tok = res.usage_metadata.prompt_token_count if res.usage_metadata else 0
                c_tok = res.usage_metadata.candidates_token_count if res.usage_metadata else 0
                return txt, p_tok, c_tok
            except Exception as e:
                print(f"[Page OCR] Gemini Vision failed: {e}", flush=True)

    return "", 0, 0


def ocr_whole_pdf(file_path: str, engine: Optional[str] = None) -> dict:
    """
    Renders every page of the PDF at 200 DPI and extracts clean Unicode text
    via Vision / Tesseract OCR when consecutive unreadable questions are detected.

    Returns:
        {
            "full_text": str,
            "pages": list[dict],
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int,
        }
    """
    doc = pymupdf.open(file_path)
    full_pages = []
    pages_metadata = []
    total_p_tok = 0
    total_c_tok = 0

    print(f"[Whole PDF OCR] Triggered full document OCR scan on {len(doc)} pages: {file_path}", flush=True)

    for page_idx, page in enumerate(doc):
        print(f"[Whole PDF OCR] Processing page {page_idx + 1}/{len(doc)}...", flush=True)
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")

        txt, p_tok, c_tok = ocr_page_vision(img_bytes, engine=engine, force=True)
        total_p_tok += p_tok
        total_c_tok += c_tok

        from .text_cleaner import clean_indic_text
        cleaned_txt, _ = clean_indic_text(txt) if txt else ("", [])
        final_page_text = cleaned_txt or txt or ""

        full_pages.append(final_page_text)
        pages_metadata.append({
            "page_num": page_idx,
            "text": final_page_text,
            "font_hashes": {},
            "char_confidences": {},
            "unmapped_codes": [],
            "validation_flags": [],
            "has_text_layer": True,
            "is_ocr_scanned": True,
        })

    doc.close()
    full_text = "\n\n".join(full_pages)
    print(f"[Whole PDF OCR] Completed full document scan. Total text extracted: {len(full_text)} chars", flush=True)

    return {
        "full_text": full_text,
        "pages": pages_metadata,
        "prompt_tokens": total_p_tok,
        "completion_tokens": total_c_tok,
        "total_tokens": total_p_tok + total_c_tok,
    }


def _rebuild_text_from_font_map(page, font_map: dict, equation_items: list[dict] = None) -> tuple[str, list[dict]]:
    r"""
    Rebuilds page text non-destructively using 2D spatial reading order and mathematical equation reconstruction.

    Returns:
        (page_text, unmapped_codes) where unmapped_codes is a list of
        {"code": int, "font": str, "bbox": list} for every character
        that could not be resolved through the font map.

    Behavior:
    - Preserves all already-valid Malayalam Unicode characters.
    - Preserves natural spacing and Latin text (headers, English terms).
    - Substitutes unmapped ligatures/symbols from the resolved Indic font map.
    - Synthesizes mathematical fractions from horizontal vector drawing bars (page.get_drawings())
      and spatial numerator/denominator span clusters into LaTeX \frac{...}{...}.
    - Formats subscripts (_{...}) and superscripts (^{...}) based on font metrics and vertical displacement.
    - Normalizes radical symbols (√) into LaTeX \sqrt{...} and Unicode minus into standard hyphens.
    - Spatially interleaves inline math/equation images based on bounding box alignment.
    - Groups text into horizontal visual bands and sorts elements left-to-right (x0).
    - Displays diagrams and figures as centered block elements.
    """
    rawdict = page.get_text("rawdict")
    unmapped_codes = []
    eq_items = list(equation_items) if equation_items else []

    # ------------------------------------------------------------------
    # Step 0: Extract candidate fraction division bars from vector drawings
    # ------------------------------------------------------------------
    candidate_bars = []
    try:
        drawings = page.get_drawings()
        for d in drawings:
            r = d.get("rect")
            if r and r.height <= 2.5 and 3.5 <= r.width <= 320.0:
                candidate_bars.append(r)
        candidate_bars.sort(key=lambda r: (r.y0, r.x0))
    except Exception:
        candidate_bars = []

    # ------------------------------------------------------------------
    # Step 1: Pre-process and resolve all glyph spans across blocks and lines
    # ------------------------------------------------------------------
    all_spans = []
    for bi, b in enumerate(rawdict.get("blocks", [])):
        if "lines" not in b:
            continue
        for li, l in enumerate(b["lines"]):
            for si, s in enumerate(l["spans"]):
                span_font = s["font"]
                base_font = span_font.split("+")[-1]
                font_size = s["size"]
                f_map = font_map.get(span_font) or font_map.get(base_font) if font_map else None
                is_legacy_font = "ML-TT" in span_font or "FML" in span_font
                flags = s.get("flags", 0)

                resolved_chars = []
                for c in s.get("chars", []):
                    ch = c["c"]
                    code = ord(ch) if len(ch) == 1 else 0
                    bbox = c.get("bbox")

                    w = (bbox[2] - bbox[0]) if bbox else 0
                    is_real_glyph = (
                        not c.get("synthetic")
                        and f_map
                        and (str(code) in f_map or code in f_map)
                        and font_size > 0
                        and (w / font_size) > 0.45
                    )

                    if (ch in ("\n", "\r", "\t", " ", "\xa0") or c.get("synthetic")) and not is_real_glyph:
                        resolved = ch
                    elif 0x0D00 <= code <= 0x0D7F:
                        resolved = ch
                    elif 32 <= code <= 126 and not is_legacy_font and re.match(r'^[a-zA-Z0-9\s\!\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~]+$', ch):
                        resolved = ch
                    elif f_map and (str(code) in f_map or code in f_map):
                        resolved = f_map.get(str(code)) or f_map.get(code)
                    else:
                        resolved = ch
                        if f_map and code > 32 and ch not in (' ', '\n', '\t', '\r', '\xa0'):
                            unmapped_codes.append({
                                "code": code,
                                "font": span_font,
                                "bbox": list(bbox) if bbox else None
                            })

                    resolved_chars.append({
                        "ch": resolved,
                        "bbox": bbox,
                        "code": code
                    })

                span_text = "".join(rc["ch"] for rc in resolved_chars)
                if not span_text.strip():
                    continue

                all_spans.append({
                    "id": f"{bi}_{li}_{si}",
                    "bi": bi,
                    "li": li,
                    "si": si,
                    "text": span_text,
                    "bbox": list(s["bbox"]),
                    "size": font_size,
                    "flags": flags,
                    "font": span_font,
                    "chars": resolved_chars
                })

    POST_BASE_SIGNS = (
        0x0D02, 0x0D03, 0x0D3E, 0x0D3F, 0x0D40, 0x0D41, 0x0D42, 0x0D43, 0x0D44,
        0x0D4A, 0x0D4B, 0x0D4C, 0x0D4D, 0x0D57
    )

    def format_span_tokens(span_list):
        if not span_list:
            return ""
        out = []
        base_y1 = None
        base_sz = None
        prev_x1 = None

        for s in span_list:
            t = s["text"]
            sz = s["size"]
            bb = s["bbox"]
            fl = s.get("flags", 0)

            is_combining = (
                bool(t) and (
                    (ord(t[0]) in POST_BASE_SIGNS) or
                    t.startswith("\u0D4D")
                )
            )

            is_delim = t.strip() in ("]", ")", "}", "[", "(", "{", ",", ";", ":", "|")

            if prev_x1 is not None and bb:
                gap = bb[0] - prev_x1
                if gap > sz * 0.22 and not is_combining:
                    prev_text = "".join(out)
                    if not prev_text.endswith(" ") and not t.startswith(" ") and not is_delim:
                        out.append(" ")

            formatted = t
            if base_y1 is not None and base_sz is not None and not is_delim:
                if sz < base_sz * 0.88 and bb[3] > base_y1 + 0.8:
                    clean_sub = t.strip()
                    if clean_sub:
                        formatted = f"_{{{clean_sub}}}" if len(clean_sub) > 1 or not clean_sub.isalnum() else f"_{clean_sub}"
                elif (fl & 1) or (sz < base_sz * 0.88 and bb[3] < base_y1 - 0.8):
                    clean_sup = t.strip()
                    if clean_sup:
                        formatted = f"^{{{clean_sup}}}" if len(clean_sup) > 1 or not clean_sup.isalnum() else f"^{clean_sup}"

            out.append(formatted)
            if not is_delim and sz >= 6.0:
                base_y1 = bb[3]
                base_sz = sz
            if bb:
                prev_x1 = bb[2]

        res = "".join(out)
        res = re.sub(r'√\(([^\)]+)\)', r'\\sqrt{\1}', res)
        res = re.sub(r'√([a-zA-Z0-9]+(?:_\{?[a-zA-Z0-9]+\}?)?)', r'\\sqrt{\1}', res)
        res = res.replace("−", "-")
        res = res.replace('\u2044', '/').replace('\u2215', '/')
        return res

    # ------------------------------------------------------------------
    # Step 2: Match fraction division bars with numerator & denominator
    # ------------------------------------------------------------------
    used_span_ids = set()
    fraction_items = []

    for bar in candidate_bars:
        y_bar = (bar.y0 + bar.y1) / 2
        num_spans = []
        den_spans = []

        for s in all_spans:
            if s["id"] in used_span_ids:
                continue
            sb = s["bbox"]
            h_overlap = min(sb[2], bar.x1 + 2.0) - max(sb[0], bar.x0 - 2.0)
            if h_overlap <= 0:
                continue
            span_w = sb[2] - sb[0]
            if span_w > 0 and (h_overlap / span_w) < 0.30:
                continue

            if y_bar - 11.5 <= sb[3] <= y_bar + 0.8:
                num_spans.append(s)
            elif y_bar - 0.8 <= sb[1] <= y_bar + 11.5:
                den_spans.append(s)

        if not num_spans or not den_spans:
            continue

        num_spans.sort(key=lambda s: s["bbox"][0])
        den_spans.sort(key=lambda s: s["bbox"][0])
        num_str = format_span_tokens(num_spans).strip()
        den_str = format_span_tokens(den_spans).strip()

        if not num_str or not den_str:
            continue
        if "√" in den_str and bar.width < 10 and len(num_str) > 2:
            continue

        for s in num_spans + den_spans:
            used_span_ids.add(s["id"])

        all_bb = [s["bbox"] for s in num_spans + den_spans]
        min_y = min(b[1] for b in all_bb)
        max_y = max(b[3] for b in all_bb)

        fraction_items.append({
            "text": f"\\frac{{{num_str}}}{{{den_str}}}",
            "bbox": [bar.x0, min_y, bar.x1, max_y],
            "y_center": y_bar,
            "x0": bar.x0,
            "x1": bar.x1,
            "y0": min_y,
            "y1": max_y,
            "is_block": False,
            "is_fraction": True
        })

    # ------------------------------------------------------------------
    # Step 3: Segment remaining text spans in each line around fractions
    # ------------------------------------------------------------------
    segmented_lines = []
    for bi, b in enumerate(rawdict.get("blocks", [])):
        if "lines" not in b:
            continue
        for li, l in enumerate(b["lines"]):
            current_seg = []
            for si, s in enumerate(l["spans"]):
                sid = f"{bi}_{li}_{si}"
                matching_all = [sp for sp in all_spans if sp["id"] == sid]
                if not matching_all:
                    continue
                matched_span = matching_all[0]

                if sid in used_span_ids:
                    if current_seg:
                        segmented_lines.append(current_seg)
                        current_seg = []
                else:
                    current_seg.append(matched_span)
            if current_seg:
                segmented_lines.append(current_seg)

    text_lines = []
    for seg in segmented_lines:
        seg_str = format_span_tokens(seg).strip()
        if not seg_str:
            continue
        min_x = min(s["bbox"][0] for s in seg)
        max_x = max(s["bbox"][2] for s in seg)
        min_y = min(s["bbox"][1] for s in seg)
        max_y = max(s["bbox"][3] for s in seg)
        y_cent = (min_y + max_y) / 2
        text_lines.append({
            "text": seg_str,
            "spans": seg,
            "bbox": [min_x, min_y, max_x, max_y],
            "y_center": y_cent,
            "x0": min_x,
            "x1": max_x,
            "y0": min_y,
            "y1": max_y,
            "is_block": False
        })

    if not text_lines and not fraction_items and not eq_items:
        return "", unmapped_codes

    # ------------------------------------------------------------------
    # Step 4: Top-to-bottom reading order reconstruction (with column awareness)
    # ------------------------------------------------------------------
    raw_elements = text_lines + fraction_items + eq_items
    page_width = page.rect.width
    mid_x = page_width / 2.0

    def render_elements_top_to_bottom(elements: list[dict]) -> list[str]:
        if not elements:
            return []

        # Sort elements primarily by vertical position, secondarily by x0
        elements.sort(key=lambda item: (item["y_center"], item["x0"]))

        # Group inline elements into vertical lines while keeping block diagrams standalone
        bands = []
        for elem in elements:
            if elem.get("is_block"):
                bands.append({
                    "avg_y": elem["y_center"],
                    "y0": elem["y0"],
                    "y1": elem["y1"],
                    "lines": [elem],
                    "is_block": True
                })
                continue

            placed = False
            for band in bands:
                if band.get("is_block"):
                    continue
                v_overlap = max(0, min(elem["y1"], band["y1"]) - max(elem["y0"], band["y0"]))
                elem_h = max(1.0, elem["y1"] - elem["y0"])
                band_h = max(1.0, band["y1"] - band["y0"])
                if abs(elem["y_center"] - band["avg_y"]) <= 4.0 or v_overlap >= 0.45 * min(elem_h, band_h):
                    band["lines"].append(elem)
                    band["avg_y"] = sum(l["y_center"] for l in band["lines"]) / len(band["lines"])
                    band["y0"] = min(band["y0"], elem["y0"])
                    band["y1"] = max(band["y1"], elem["y1"])
                    placed = True
                    break

            if not placed:
                bands.append({
                    "avg_y": elem["y_center"],
                    "y0": elem["y0"],
                    "y1": elem["y1"],
                    "lines": [elem],
                    "is_block": False
                })

        # Sort lines strictly top to bottom by y0
        bands.sort(key=lambda b: b["y0"])

        rendered = []
        for band in bands:
            if band.get("is_block"):
                rendered.append(band["lines"][0]["text"])
                continue

            # Check if non-fraction text lines in this band overlap horizontally
            text_lines_in_band = [l for l in band["lines"] if not l.get("is_fraction") and "spans" in l]
            has_overlap = False
            if len(text_lines_in_band) > 1:
                for i in range(len(text_lines_in_band)):
                    for j in range(i + 1, len(text_lines_in_band)):
                        if (text_lines_in_band[i]["x0"] < text_lines_in_band[j]["x1"] - 1.0 and
                            text_lines_in_band[j]["x0"] < text_lines_in_band[i]["x1"] - 1.0):
                            has_overlap = True
                            break
                    if has_overlap:
                        break

            if has_overlap and len(band["lines"]) == len(text_lines_in_band):
                all_band_spans = []
                for l in text_lines_in_band:
                    all_band_spans.extend(l["spans"])

                non_ws = [s for s in all_band_spans if s.get("text", "").strip()]
                filtered_spans = []
                for s in all_band_spans:
                    st = s.get("text", "")
                    if not st.strip():
                        sb = s.get("bbox", [0, 0, 0, 0])
                        if any(not (sb[2] <= nws["bbox"][0] or nws["bbox"][2] <= sb[0]) for nws in non_ws):
                            continue
                    filtered_spans.append(s)

                filtered_spans.sort(key=lambda s: s["bbox"][0])
                merged_str = format_span_tokens(filtered_spans).strip()
                merged_str = re.sub(r' +', ' ', merged_str).strip()
                if merged_str:
                    rendered.append(merged_str)
                continue

            band["lines"].sort(key=lambda l: l["x0"])
            band_str = ""
            prev_x1 = None

            for l in band["lines"]:
                text = l["text"]
                if not text:
                    continue

                if prev_x1 is not None:
                    gap = l["x0"] - prev_x1
                    is_opt_start = bool(re.match(r'^\s*(?:\([A-Ea-e1-4]\)|\[[A-Ea-e1-4]\]|[A-Ea-e][\.\)\:-](?!\S)|[1-4]\.(?!\d)\s+)', text))
                    if is_opt_start:
                        if not band_str.endswith("\t") and not band_str.endswith("   "):
                            band_str += " \t "
                    elif gap > 25:
                        band_str += " \t "
                    elif gap > 1.5:
                        if not band_str.endswith(" ") and not text.startswith(" ") and not text.lstrip().startswith((",", ".", ";", ":", ")", "]", "}")):
                            band_str += " "

                band_str += text
                prev_x1 = max(prev_x1 if prev_x1 is not None else 0, l["x1"])

            band_str = re.sub(r' +', ' ', band_str).strip()
            if band_str:
                rendered.append(band_str)

        return rendered

    # Detect 2-column layout:
    left_items = [e for e in raw_elements if e["x1"] <= mid_x + 10]
    right_items = [e for e in raw_elements if e["x0"] >= mid_x - 10]
    crossing_items = [e for e in raw_elements if e["x0"] < mid_x - 15 and e["x1"] > mid_x + 15]

    is_two_column = (
        len(left_items) >= 6 and
        len(right_items) >= 6 and
        len(crossing_items) <= 0.25 * len(raw_elements)
    )

    if is_two_column:
        col_top = min(min(e["y0"] for e in left_items), min(e["y0"] for e in right_items))
        col_bottom = max(max(e["y1"] for e in left_items), max(e["y1"] for e in right_items))

        headers = [e for e in raw_elements if e["y1"] <= col_top + 5 or (e["x0"] < mid_x - 15 and e["x1"] > mid_x + 15 and e["y0"] < col_top + 40)]
        footers = [e for e in raw_elements if e["y0"] >= col_bottom - 5 or (e["x0"] < mid_x - 15 and e["x1"] > mid_x + 15 and e["y1"] > col_bottom - 40)]
        header_ids = {id(e) for e in headers}
        footer_ids = {id(e) for e in footers}

        col_left = [e for e in left_items if id(e) not in header_ids and id(e) not in footer_ids]
        col_right = [e for e in right_items if id(e) not in header_ids and id(e) not in footer_ids]

        page_lines = []
        page_lines.extend(render_elements_top_to_bottom(headers))
        page_lines.extend(render_elements_top_to_bottom(col_left))
        page_lines.extend(render_elements_top_to_bottom(col_right))
        page_lines.extend(render_elements_top_to_bottom(footers))
    else:
        page_lines = render_elements_top_to_bottom(raw_elements)

    return "\n".join(page_lines), unmapped_codes

def extract_text_from_pdf(file_path: str) -> dict:
    """
    Universal PDF text extraction pipeline.
    
    Returns a rich dict with:
        - full_text: concatenated page text
        - pages: per-page metadata (text, font_hashes, unmapped_codes, has_text_layer)
        - prompt_tokens / completion_tokens: token counters (now only from
          the FML converter or future repair_node, NOT from page-level OCR fallback)

    Per page:
    1. Detect inline math/equation images and diagrams, validate crops, preserve as crisp base64 PNGs
    2. Detect subset fonts → build/retrieve font maps
    3. Rebuild text using font maps with exact 2D spatial reading order interleaving
    4. Clean and normalize Indic text (with HTML image tag protection)
    
    NOTE: Page-level sanity check + full-page Vision OCR fallback has been REMOVED.
    Quality scoring and repair are handled per-question in the graph pipeline.
    """
    from .quality_check import validate_equation_crop
    from .font_mapper import get_or_create_font_map, hash_font_data

    doc = pymupdf.open(file_path)
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    
    full_pages = []
    pages_metadata = []
    all_span_blocks: list[dict] = []   # layout-aware span stream for layout_segmenter
    total_p_tokens = 0
    total_c_tokens = 0

    for page_idx, page in enumerate(doc):
        images = page.get_images()

        # Give every diagram an immutable source owner before text is rebuilt.
        # Text-order heuristics alone can otherwise assign a lower-page figure to
        # the preceding question when columns, answers, or page breaks are present.
        question_anchors = []
        for text_block in page.get_text("blocks"):
            if len(text_block) < 5:
                continue
            header = text_block[4].strip()
            match = re.match(
                r'^(?:Question\s*|Q\.?\s*)?(\d{1,3})\s*[\.\):\-]',
                header,
                re.IGNORECASE,
            )
            if match:
                question_anchors.append((text_block[1], int(match.group(1))))
        question_anchors.sort()

        # ---------------------------------------------------------------
        # Step 1: Pre-extract equation clips and diagrams on the page
        # ---------------------------------------------------------------
        equation_items = []
        seen_rect_keys = set()
        if images:
            for img_info in images:
                xref = img_info[0]
                rects = page.get_image_rects(xref)
                if not rects:
                    continue
                for r in rects:
                    r_key = (round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1))
                    if r_key in seen_rect_keys:
                        continue
                    seen_rect_keys.add(r_key)

                    # Skip full-page background images
                    if r.width > page.rect.width * 0.85 and r.height > page.rect.height * 0.5:
                        continue
                    # Skip tiny speckles
                    if r.width < 3 or r.height < 3:
                        continue

                    # Render sharp high-DPI image clip (200 DPI gives excellent clarity while staying lightweight)
                    strip_pix = page.get_pixmap(clip=r, dpi=200)
                    crop_bytes = strip_pix.tobytes("png")

                    # Validate equation crop before splicing into text
                    if not validate_equation_crop(crop_bytes, (r.x0, r.y0, r.x1, r.y1)):
                        print(f"Page {page_idx + 1}: Dropping invalid equation crop at ({r.x0:.0f},{r.y0:.0f})-({r.x1:.0f},{r.y1:.0f})", flush=True)
                        continue

                    h = r.height
                    w = r.width
                    # Distinguish block diagrams/figures from inline mathematical expressions
                    is_block = (h > 75 or (h > 55 and w > 280))

                    # Try OCR first for equations/math snippets, but SKIP for block diagrams
                    if not is_block:
                        math_text, p_tok, c_tok = ocr_equation_clip(crop_bytes)
                        if math_text:
                            total_p_tokens += p_tok
                            total_c_tokens += c_tok
                            # Emit as {{EQ:...}} anchor token — expand_anchor_tokens() resolves
                            # it to inline LaTeX at the API serialisation layer.  This prevents
                            # the raw LaTeX from being mistaken for prose by parse_questions_from_text.
                            eq_token = f"{{{{EQ:{math_text}}}}}"
                            equation_items.append({
                                "text": f" {eq_token} ",
                                "rect": r,
                                "bbox": [r.x0, r.y0, r.x1, r.y1],
                                "x0": r.x0,
                                "x1": r.x1,
                                "y0": r.y0,
                                "y1": r.y1,
                                "y_center": (r.y0 + r.y1) / 2,
                                "is_block": False,
                                # span_blocks metadata
                                "type": "equation",
                                "page": page_idx,
                                "font_size": 0.0,
                                "is_bold": False,
                                "x_indent": r.x0 / max(page.rect.width, 1),
                            })
                            continue

                    img_url = upload_image_to_cloudflare(crop_bytes, filename=base_filename)
                    owner_number = next(
                        (number for y_pos, number in reversed(question_anchors) if y_pos <= r.y0 + 2),
                        None,
                    )

                    # Emit as anchor token — expand_anchor_tokens() builds the final HTML.
                    # This fixes the long-standing bug where an img tag's closing "/>" was
                    # mistaken for trailing math punctuation by parse_questions_from_text,
                    # causing the next question's boundary to be silently rejected.
                    img_token = _make_img_anchor(img_url, h, w, is_block, page_idx, owner_number)

                    equation_items.append({
                        "text": f" {img_token} " if not is_block else img_token,
                        "rect": r,
                        "bbox": [r.x0, r.y0, r.x1, r.y1],
                        "x0": r.x0,
                        "x1": r.x1,
                        "y0": r.y0,
                        "y1": r.y1,
                        "y_center": (r.y0 + r.y1) / 2,
                        "is_block": is_block,
                        # span_blocks metadata
                        "type": "image",
                        "page": page_idx,
                        "font_size": 0.0,
                        "is_bold": False,
                        "x_indent": r.x0 / max(page.rect.width, 1),
                    })

        # ---------------------------------------------------------------
        # Step 2: Universal font map detection and text rebuild
        # ---------------------------------------------------------------
        page_blocks = [b[4] for b in page.get_text("blocks") if len(b) > 4 and b[4].strip()]
        font_map, confidence_map = get_or_create_font_map(doc, page_idx, affected_texts=page_blocks)

        # Collect font hashes for this page (for downstream repair_node)
        page_font_hashes = {}
        page_fonts = page.get_fonts(full=True)
        for fi in page_fonts:
            xref = fi[0]
            font_name = fi[3]
            try:
                _, _, _, buf = doc.extract_font(xref)
                if buf:
                    page_font_hashes[font_name] = hash_font_data(buf)
            except Exception:
                pass

        # Noto Malayalam embeds a valid Unicode text layer. Rebuilding those
        # glyphs through the universal shape mapper can reinterpret real spaces
        # and punctuation as legacy glyphs, so keep the native Unicode stream.
        has_noto_malayalam = any(
            "NotoSansMalayalam" in (font[3] or "")
            for font in page_fonts
        )
        if font_map and not has_noto_malayalam:
            font_names = list(set(font_map.keys()))
            print(f"Page {page_idx + 1}: {len(font_names)} mapped fonts. Rebuilding text from CIDs...", flush=True)
            page_text, unmapped_codes = _rebuild_text_from_font_map(page, font_map, equation_items)
        else:
            page_text, unmapped_codes = _rebuild_text_from_font_map(page, {}, equation_items)

        # ---------------------------------------------------------------
        # Step 3: Clean and normalize
        # ---------------------------------------------------------------
        from .text_cleaner import clean_indic_text
        page_text, page_flags = clean_indic_text(page_text)

        # ---------------------------------------------------------------
        # Step 4: FML/ML-TT legacy font heuristic
        # ---------------------------------------------------------------
        malayalam_chars_count = len(re.findall(r'[\u0D00-\u0D7F]', page_text))
        has_fml_font = any(
            any(legacy in (fi[3] or "") for legacy in ("ML-TT", "FML", "Karthika", "Revathi", "Ambili", "Panchari"))
            for fi in page_fonts
        )
        if has_fml_font and malayalam_chars_count < 20 and re.search(r'[ß∂Ø∏ƒ≥‰≠‡∑›™‚]', page_text):
            print(f"Page {page_idx + 1}: FML/ML-TT legacy symbols detected in font {page_fonts}. Transliterating...", flush=True)
            from .smc_fml_converter import convert_fml_to_unicode
            page_text = convert_fml_to_unicode(page_text)
            page_text, extra_flags = clean_indic_text(page_text)
            page_flags.extend(extra_flags)

        # ---------------------------------------------------------------
        # Step 5: Check for image-only pages (no text layer at all)
        # ---------------------------------------------------------------
        has_text_layer = bool(page.get_text("text").strip())

        if not page_text.strip() and images:
            print(f"Page {page_idx + 1}: No text, {len(images)} images. Will be handled by repair_node if needed.", flush=True)
            # Still attempt page Vision OCR for image-only pages as a baseline
            page_pix = page.get_pixmap(dpi=200)
            fallback_text, p_tok, c_tok = ocr_page_vision(page_pix.tobytes("png"))
            total_p_tokens += p_tok
            total_c_tokens += c_tok
            if fallback_text:
                page_text = fallback_text

        # Kept outside question text but lets the parser retain a physical page
        # identity for reconciliation of diagram ownership.
        full_pages.append(f"<!--QAI_PAGE:{page_idx}-->\n{page_text}")

        # Store per-page metadata for downstream quality scoring and repair
        pages_metadata.append({
            "page_num": page_idx,
            "text": page_text,
            "font_hashes": page_font_hashes,
            "char_confidences": confidence_map,  # per-glyph resolution confidence for word routing
            "unmapped_codes": unmapped_codes,
            "validation_flags": page_flags,
            "has_text_layer": has_text_layer,
        })

        # ---------------------------------------------------------------
        # Step 6: Collect span_blocks for the layout-aware segmenter
        # ---------------------------------------------------------------
        page_width = max(page.rect.width, 1)
        
        # Extract candidate fraction division bars from vector drawings
        candidate_bars = []
        try:
            drawings = page.get_drawings()
            for d in drawings:
                r = d.get("rect")
                if r and r.height <= 2.5 and 3.5 <= r.width <= 320.0:
                    candidate_bars.append(r)
            candidate_bars.sort(key=lambda r: (r.y0, r.x0))
        except Exception:
            candidate_bars = []

        # Add drawing paths (horizontal lines) to span_blocks for fraction detection
        for bar in candidate_bars:
            all_span_blocks.append({
                "type": "line",
                "text": "—",
                "bbox": [bar.x0, bar.y0, bar.x1, bar.y1],
                "page": page_idx,
                "font_size": 0.0,
                "is_bold": False,
                "x_indent": bar.x0 / page_width,
            })
            
        for eq_item in equation_items:
            # equation_items already carry the span_blocks metadata fields
            # added above in Step 1 during this iteration; collect them.
            if "type" in eq_item:
                all_span_blocks.append({
                    "type": eq_item["type"],
                    "text": eq_item["text"].strip(),
                    "bbox": eq_item["bbox"],
                    "page": page_idx,
                    "font_size": eq_item.get("font_size", 0.0),
                    "is_bold": eq_item.get("is_bold", False),
                    "x_indent": eq_item.get("x_indent", 0.0),
                })

        # Collect text spans: iterate all_spans (populated in _rebuild_text_from_font_map
        # via the rawdict pass) and emit one span_block per span that was not consumed
        # by the fraction-bar matcher.  We use the flags/size/bbox already on each span.
        # all_spans is a local in _rebuild_text_from_font_map so we re-extract the
        # page rawdict here at minimal cost (just the dict, no re-rendering).
        try:
            dict_spans = page.get_text("dict", flags=0)
            for blk in dict_spans.get("blocks", []):
                if "lines" not in blk:
                    continue
                for ln in blk["lines"]:
                    for sp in ln["spans"]:
                        sp_text = sp.get("text", "").strip()
                        if not sp_text:
                            continue
                        sp_bbox = list(sp.get("bbox", [0, 0, 0, 0]))
                        sp_size = sp.get("size", 0.0)
                        sp_flags = sp.get("flags", 0)
                        sp_x0 = sp_bbox[0] if sp_bbox else 0.0
                        sp_font = sp.get("font", "")
                        all_span_blocks.append({
                            "type": "text",
                            "text": sp_text,
                            "bbox": sp_bbox,
                            "page": page_idx,
                            "font_size": sp_size,
                            "is_bold": bool(sp_flags & 16),
                            "flags": sp_flags,
                            "font": sp_font,
                            "x_indent": sp_x0 / page_width,
                        })
        except Exception as _span_err:
            print(f"[span_blocks] Page {page_idx + 1}: could not extract span metadata: {_span_err}", flush=True)

    # Sort span blocks in reading order so images (which are extracted first)
    # and line-spans are correctly interleaved vertically and horizontally with text.
    from .layout_segmenter import sort_blocks_in_reading_order
    all_span_blocks = sort_blocks_in_reading_order(all_span_blocks)

    return {
        "full_text": "\n\n".join(full_pages),
        "pages": pages_metadata,
        "span_blocks": all_span_blocks,
        "prompt_tokens": total_p_tokens,
        "completion_tokens": total_c_tokens,
    }


def route_and_extract(file_path: str) -> dict:
    """Ingestion routing layer entrypoint."""
    return extract_text_from_pdf(file_path)
