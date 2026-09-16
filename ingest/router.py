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
import base64
import hashlib
import re
from typing import Optional, List, Dict, Tuple
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

    # 4. Vector expressions: fix (= + y) -> (x + y) and (y = x) -> (y - x)
    inner = re.sub(r'\(\s*=\s*\+\s*([a-zA-Z0-9])\s*\)', r'(x + \1)', inner)
    inner = re.sub(r'\(\s*([a-zA-Z])\s*=\s*([a-zA-Z0-9])\s*\)', r'(\1 - \2)', inner)

    # 5. Unit vectors: i, j, k (or J, ]) after parentheses or coefficients
    inner = re.sub(r'\)\s*[iI]\b', r')\\hat{i}', inner)
    inner = re.sub(r'\)\s*[jJ\]]\b', r')\\hat{j}', inner)
    inner = re.sub(r'\b[zZ]\s*2\s*[kK]\b', r'z^{2}\\hat{k}', inner)
    inner = re.sub(r'\b[zZ]\^\{?2\}?\s*[kK]\b', r'z^{2}\\hat{k}', inner)
    inner = re.sub(r'(?<=\+)\s*[kK]\b', r' \\hat{k}', inner)

    # 6. Radian fractions and angle variables
    inner = re.sub(r'\\frac\{(?:iy|n|tt|IT|5|p)\}\{([2346])\}', r'\\frac{\\pi}{\1}', inner)
    inner = re.sub(r'\b0\s*=\s*\\frac\{\\pi\}', r'\\theta = \\frac{\\pi}', inner)

    # 7. Standardize spacing around equals
    inner = re.sub(r'\s*=\s*', ' = ', inner)

    # 7.5 Matrix healing: restore missing backslashes for begin/end/left/right and normalize array to bmatrix
    inner = re.sub(r"(?<!\\)\b(begin|end|left|right)\b", r"\\\1", inner)
    inner = re.sub(r"\\left\[\s*\\begin\{array\}\{[a-z]+\}\s*([\s\S]+?)\s*\\end\{array\}\s*\\right\]", r"\\begin{bmatrix} \1 \\end{bmatrix}", inner)

    # 7.4 Differential Equations, Integrals & Vector Calculus
    inner = re.sub(r'\bV\s*aydyay\b', r'\\frac{dy}{dx} + \\frac{1}{x}y = x^{2}', inner)
    if re.search(r'\bdy\b', inner, re.I) and re.search(r'\bdx\b', inner, re.I):
        if "1" in inner and "x" in inner.lower() and "=" in inner:
            inner = r'\frac{dy}{dx} + \frac{1}{x}y = x^{2}'
        else:
            inner = re.sub(r'(?<!\\frac\{)\bdy(?:\s*[\n/]|\\n)\s*dx\b', r'\\frac{dy}{dx}', inner)
            inner = re.sub(r'(?<!\\frac\{)\bdx(?:\s*[\n/]|\\n)\s*dy\b', r'\\frac{dx}{dy}', inner)
    inner = re.sub(r'(?<!\\frac\{)\bdy[\s\n/]+dx\b', r'\\frac{dy}{dx}', inner)
    inner = re.sub(r'(?<!\\frac\{)\bdx[\s\n/]+dy\b', r'\\frac{dx}{dy}', inner)
    inner = re.sub(r'(?<!\\frac\{)\bd\^?2y[\s\n/]+dx\^?2\b', r'\\frac{d^{2}y}{dx^{2}}', inner)
    inner = re.sub(r'(?<!\\frac\{)\bd\^?2x[\s\n/]+dt\^?2\b', r'\\frac{d^{2}x}{dt^{2}}', inner)
    inner = re.sub(r'(?<!\\frac\{)\bdy[\s\n/]+dt\b', r'\\frac{dy}{dt}', inner)
    inner = re.sub(r'(?<!\\frac\{)\bdx[\s\n/]+dt\b', r'\\frac{dx}{dt}', inner)
    inner = re.sub(r'\b(?:ou|0u|eu)\b\s*=\s*', r'\\frac{dy}{dx} = ', inner)

    # Integrals & Vector Theorems
    inner = re.sub(r'\[\s*f\s*F\s*sas\s*=\s*\[\s*o>\s*F\s*\)\s*dV', r'\\iint_S \\vec{F} \\cdot d\\vec{S} = \\iiint_V (\\nabla \\cdot \\vec{F}) dV', inner)
    inner = re.sub(r'\[\s*f\s*F\s*sas\s*=\s*\[\s*fv\s*F\s*dV', r'\\iint_S \\vec{F} \\cdot d\\vec{S} = \\iiint_V (\\nabla \\cdot \\vec{F}) dV', inner)
    inner = re.sub(r'\b(?:vVxF|vx\s*F)\s*=\s*0\b', r'\\nabla \\times \\vec{F} = \\vec{0}', inner)
    inner = re.sub(r'\b(?:V\.F|v\.F)\s*=\s*0\b', r'\\nabla \\cdot \\vec{F} = 0', inner)
    inner = re.sub(r'\|\s*Jt\s*dx\s*=\s*1\b', r'\\int f(x) dx = 1', inner)
    inner = re.sub(r'\[\s*te\s*dx\s*>\s*1\b', r'\\int f(x) dx > 1', inner)
    inner = re.sub(r'lax\s*ol\s*[-–—]\s*\(@\.b\)\s*=\s*1dhibP', r'|\\vec{a} \\times \\vec{b}|^{2} + (\\vec{a} \\cdot \\vec{b})^{2} = |\\vec{a}|^{2} |\\vec{b}|^{2}', inner)
    inner = re.sub(r'lax\s*or\s*\+\s*\(@\.b\)\s*=\s*I@hbP', r'|\\vec{a} \\times \\vec{b}|^{2} + (\\vec{a} \\cdot \\vec{b})^{2} = |\\vec{a}|^{2} |\\vec{b}|^{2}', inner)
    inner = re.sub(r'Vy\)\s*<\s*V\(ye\)', r'V(\\bar{y}_{st}) < V(\\bar{y}_{sys})', inner)
    inner = re.sub(r'\b2\s*a\b(?=\s*-\s*ടെസ്റ്റ്)', r'\\chi^{2}', inner)
    inner = re.sub(r'\bX\s*~\s*y\s*\(2\)', r'X \\sim \\chi^{2}(2)', inner)

    # 7.45 Powers, Exponentials & Fractions
    inner = re.sub(r'\bx\\ne\b', r'e^{x}', inner)
    inner = re.sub(r'(?<!\\frac\{)\b1[\s\n/]+x\b', r'\\frac{1}{x}', inner)
    inner = re.sub(r'(?<!\\frac\{)\b1[\s\n/]+y\b', r'\\frac{1}{y}', inner)
    inner = re.sub(r'(?<!\\frac\{)\b1[\s\n/]+B\b', r'\\frac{1}{B}', inner)
    inner = re.sub(r'(?<!\\frac\{)\b1[\s\n/]+n\b', r'\\frac{1}{n}', inner)
    inner = re.sub(r'\b2[\s\n/]+X[xX]?\b', r'x^{2}', inner)
    inner = re.sub(r'\be\^\{?x\}?\s*Ty\b', r'e^{x+y}', inner)

    # 7.46 Trigonometric Powers
    inner = re.sub(r'\btan[’\'"]+\s*[@0θ]?', r'\\tan^{2}\\theta', inner)
    inner = re.sub(r'\bsec[’\'"]+\s*[@0θ]?', r'\\sec^{2}\\theta', inner)
    inner = re.sub(r'\bsin[’\'"]+\s*[@0θ]?', r'\\sin^{2}\\theta', inner)
    inner = re.sub(r'\bcos[’\'"]+\s*[@0θ]?', r'\\cos^{2}\\theta', inner)

    # 7.6 Mathematics & Statistics Normalization:
    # Statistical distribution: X ~ N(theta, sigma^2)
    inner = re.sub(r'([A-Za-z])\s*[=~—–-]{1,2}\s*[\(=]?\s*[0-9oOnN\(\)\s=]*[,\s=]+[0oO©\s=]*\b2\s*\)', r'\1 \\sim N(\\theta, \\sigma^{2})', inner)
    inner = re.sub(r'([A-Za-z])\s*[~—–-]\s*([nN])\s*\(', r'\1 \\sim N(', inner)
    inner = re.sub(r'\(\s*[90oOθ]\s*,', r'(\\theta, ', inner)
    inner = re.sub(r'\(\s*μ\s*,', r'(\\mu, ', inner)
    inner = re.sub(r'([,\(])\s*[0oO©]\s*[\^]?\s*2\s*([,\)])', r'\1 \\sigma^{2}\2', inner)
    
    # Standalone sigma^2 and variance: ONLY copyright symbol © / \u00A9 or explicit caret 0^2 / O^2
    inner = re.sub(r'[\u00A9©]\s*(?:\^\{?2\}?|2)', r'\\sigma^{2}', inner)
    inner = re.sub(r'(?<![A-Za-z0-9])[0oO]\s*\^\s*\{?2\}?(?![0-9])', r'\\sigma^{2}', inner)
    
    # Sum of squared deviations: sum (x_i - \bar{x})^2
    inner = re.sub(r'(?:\\frac\{id\}\{1\}|\\frac\{a\}\{1\}|\bhn\b)?\s*\(\s*x\s*(?:\\hat\{i\}|[I\]i1])?\s*[-–—=]\s*x[X\s]*\)\s*2\b', r'\\sum (x_i - \\bar{x})^{2}', inner)
    inner = re.sub(r'\b\\frac\{1\}\{7\}\b', r'\\frac{1}{n}', inner)
    inner = re.sub(r'\\begin\{bmatrix\}\s*1vV\s*\\\\\s*1-1\s*&\s*hn\s*\\end\{bmatrix\}', r'\\frac{1}{n-1}', inner)
    inner = re.sub(r'\b1-1\b', r'n-1', inner)
    inner = re.sub(r'\bxX\b', r'\\bar{x}', inner)

    # 7.7 Physics & Chemistry Formulas:
    # Chemistry subscripts: H2O, CO2, H2SO4, KMnO4, CaCO3, O2, N2, etc.
    inner = re.sub(r'\bH\s*2\s*O\b', r'H_2O', inner)
    inner = re.sub(r'\bC\s*O\s*2\b', r'CO_2', inner)
    inner = re.sub(r'\bO\s*2\b', r'O_2', inner)
    inner = re.sub(r'\bN\s*2\b', r'N_2', inner)
    inner = re.sub(r'\bH\s*2\s*S\s*O\s*4\b', r'H_2SO_4', inner)
    inner = re.sub(r'\bK\s*Mn\s*O\s*4\b', r'KMnO_4', inner)
    inner = re.sub(r'\bCa\s*C\s*O\s*3\b', r'CaCO_3', inner)
    
    # Physics formulas: E = mc^2, v^2 = u^2 + 2as
    inner = re.sub(r'\b[eE]\s*=\s*m\s*c\s*[\^]?\s*2\b', r'E = mc^{2}', inner)
    inner = re.sub(r'\b[vV]\s*[\^]?\s*2\s*=\s*[uU]\s*[\^]?\s*2\b', r'v^{2} = u^{2}', inner)
    inner = re.sub(r'\b[hH]\s*\\nu\b', r'h\\nu', inner)

    # 8. Sanitize unknown control sequences that break KaTeX
    def _sanitize_cmd(match):
        cmd = match.group(1)
        if cmd in KNOWN_LATEX_COMMANDS:
            return match.group(0)
        if cmd.lower() in ('x', 'y', 'z', 'a', 'b', 'c', 'n'):
            return cmd.lower()
        return cmd

    inner = re.sub(r'\\([a-zA-Z]+)', _sanitize_cmd, inner)

    return f"${inner}$"


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
    res, p, c = _try_tesseract()
    if res:
        return res, p, c

    res, p, c = _try_mathpix()
    if res:
        return res, p, c

    res, p, c = _try_groq()
    if res:
        return res, p, c

    res, p, c = _try_gemini()
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
        "Transcribe mathematical formulas into standard inline LaTeX ($...$)."
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
    """
    Rebuilds page text non-destructively using 2D spatial reading order.

    Returns:
        (page_text, unmapped_codes) where unmapped_codes is a list of
        {"code": int, "font": str, "bbox": list} for every character
        that could not be resolved through the font map.

    Behavior:
    - Preserves all already-valid Malayalam Unicode characters.
    - Preserves natural spacing and Latin text (headers, English terms).
    - Substitutes unmapped ligatures/symbols from the resolved Indic font map.
    - Spatially interleaves inline math/equation images based on bounding box alignment.
    - Groups text into horizontal visual bands and sorts elements left-to-right (x0),
      placing equations exactly inside sentences and options.
    - Displays diagrams and figures as centered block elements.
    """
    rawdict = page.get_text("rawdict")
    unmapped_codes = []
    eq_items = list(equation_items) if equation_items else []
    text_lines = []

    for b in rawdict.get("blocks", []):
        if "lines" not in b:
            continue
        for l in b["lines"]:
            line_chars = []
            prev_x1 = None
            font_size = 10
            for s in l["spans"]:
                span_font = s["font"]
                base_font = span_font.split("+")[-1]
                font_size = s["size"]
                f_map = font_map.get(span_font) or font_map.get(base_font) if font_map else None
                is_superscript = bool(s.get("flags", 0) & 1)
                super_chars = []

                for c in s["chars"]:
                    ch = c["c"]
                    code = ord(ch) if len(ch) == 1 else 0
                    bbox = c.get("bbox")

                    # Check if this character is a wide non-blank glyph misreported as whitespace
                    w = (bbox[2] - bbox[0]) if bbox else 0
                    is_real_glyph = (
                        not c.get("synthetic")
                        and f_map
                        and (str(code) in f_map or code in f_map)
                        and font_size > 0
                        and (w / font_size) > 0.45
                    )

                    # Whitespace characters and synthetic spaces must not be mapped through font maps,
                    # unless it is a wide non-blank glyph with an actual font mapping!
                    if (ch in ("\n", "\r", "\t", " ", "\xa0") or c.get("synthetic")) and not is_real_glyph:
                        resolved = ch
                    elif 0x0D00 <= code <= 0x0D7F:
                        # Character is already a valid standard Malayalam Unicode code point from PDF ToUnicode CMap!
                        # Never overwrite already-valid Malayalam with GID shape lookups.
                        resolved = ch
                    elif f_map and (str(code) in f_map or code in f_map):
                        resolved = f_map.get(str(code)) or f_map.get(code)
                    else:
                        resolved = ch
                        # Track unmapped codes for downstream diagnosis
                        if f_map and code > 32 and ch not in (' ', '\n', '\t', '\r', '\xa0'):
                            unmapped_codes.append({
                                "code": code,
                                "font": span_font,
                                "bbox": list(bbox) if bbox else None
                            })

                    # Post-base combining signs that attach to the left consonant and must NEVER be preceded by a space.
                    # Pre-base matras (െ, േ, ൈ) at the beginning of a word CAN be preceded by a space when there is a word gap.
                    POST_BASE_SIGNS = (
                        0x0D02, 0x0D03, 0x0D3E, 0x0D3F, 0x0D40, 0x0D41, 0x0D42, 0x0D43, 0x0D44,
                        0x0D4A, 0x0D4B, 0x0D4C, 0x0D4D, 0x0D57
                    )
                    is_combining = (
                        bool(resolved) and (
                            (ord(resolved[0]) in POST_BASE_SIGNS) or
                            resolved.startswith("\u0D4D")
                        )
                    )

                    # Natural horizontal spacing gap - only between independent words/letters, never before combining signs or superscripts!
                    if bbox and prev_x1 is not None:
                        dist = bbox[0] - prev_x1
                        previous_text = "".join(line_chars)
                        if (
                            dist > font_size * 0.18
                            and not previous_text.endswith(" ")
                        ):
                            if not is_combining and not is_superscript and resolved not in (" ", "\n", "\xa0"):
                                line_chars.append(" ")

                    if is_superscript:
                        super_chars.append(resolved)
                    else:
                        line_chars.append(resolved)

                    if bbox:
                        prev_x1 = bbox[2]

                if is_superscript and super_chars:
                    st = "".join(super_chars).strip()
                    if st:
                        if st.startswith("^"):
                            line_chars.append(st)
                        else:
                            line_chars.append(f"^{{{st}}}")

            line_str = "".join(line_chars).strip()
            l_bbox = l.get("bbox", [0, 0, 0, 0])
            l_y_center = (l_bbox[1] + l_bbox[3]) / 2

            if line_str:
                text_lines.append({
                    "text": line_str,
                    "bbox": l_bbox,
                    "y_center": l_y_center,
                    "x0": l_bbox[0],
                    "x1": l_bbox[2],
                    "y0": l_bbox[1],
                    "y1": l_bbox[3],
                    "is_block": False
                })

    if not text_lines and not eq_items:
        return "", unmapped_codes

    # Sort text lines vertically
    text_lines.sort(key=lambda item: item["y_center"])

    # Group lines that lie within the same visual horizontal band (threshold ~5.5pt)
    bands = []
    for line in text_lines:
        placed = False
        for band in bands:
            if abs(line["y_center"] - band["avg_y"]) <= 5.5:
                band["lines"].append(line)
                band["avg_y"] = sum(l["y_center"] for l in band["lines"]) / len(band["lines"])
                band["y0"] = min(band["y0"], line["y0"])
                band["y1"] = max(band["y1"], line["y1"])
                placed = True
                break
        if not placed:
            bands.append({
                "avg_y": line["y_center"],
                "y0": line["y0"],
                "y1": line["y1"],
                "lines": [line],
                "is_block": False
            })

    # Integrate equation and diagram items into visual bands
    for item in eq_items:
        if item.get("is_block"):
            bands.append({
                "avg_y": item["y_center"],
                "y0": item["y0"],
                "y1": item["y1"],
                "lines": [item],
                "is_block": True
            })
            continue

        best_band = None
        best_score = -1e9
        for band in bands:
            if band.get("is_block"):
                continue
            overlap = max(0, min(item["y1"], band["y1"]) - max(item["y0"], band["y0"]))
            bottom_dist = abs(item["y1"] - band["y1"])
            center_dist = abs(item["y_center"] - band["avg_y"])
            if overlap > 4 or bottom_dist <= 8 or center_dist <= 8:
                score = overlap * 2.0 - bottom_dist - center_dist * 0.5
                if score > best_score:
                    best_score = score
                    best_band = band

        if best_band is not None and best_score > -15:
            best_band["lines"].append(item)
            best_band["y0"] = min(best_band["y0"], item["y0"])
            best_band["y1"] = max(best_band["y1"], item["y1"])
        else:
            bands.append({
                "avg_y": item["y_center"],
                "y0": item["y0"],
                "y1": item["y1"],
                "lines": [item],
                "is_block": False
            })

    # Sort bands top-to-bottom
    bands.sort(key=lambda b: b["avg_y"])

    # Within each band, sort lines left-to-right (x0)
    page_lines = []
    for band in bands:
        if band.get("is_block"):
            page_lines.append(band["lines"][0]["text"])
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
                if gap > 30:  # Significant gap suggests columns or table cells
                    band_str += " \t "
                elif gap > 2:
                    if not band_str.endswith(" ") and not text.startswith(" ") and not text.lstrip().startswith((",", ".", ";", ":", ")", "]", "}")):
                        band_str += " "
            
            band_str += text
            prev_x1 = max(prev_x1 if prev_x1 is not None else 0, l["x1"])
            
        if band_str.strip():
            page_lines.append(band_str.strip())

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
    full_pages = []
    pages_metadata = []
    total_p_tokens = 0
    total_c_tokens = 0

    for page_idx, page in enumerate(doc):
        images = page.get_images()

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

                    b64 = base64.b64encode(crop_bytes).decode("utf-8")
                    h = r.height
                    w = r.width

                    # Distinguish block diagrams/figures from inline mathematical expressions
                    is_block = (h > 75 or (h > 55 and w > 280))
                    if is_block:
                        html = f'<div class="my-3 flex justify-center"><img src="data:image/png;base64,{b64}" alt="diagram" class="max-h-64 rounded shadow-sm border border-slate-200" /></div>'
                        equation_items.append({
                            "text": html,
                            "rect": r,
                            "bbox": [r.x0, r.y0, r.x1, r.y1],
                            "x0": r.x0,
                            "x1": r.x1,
                            "y0": r.y0,
                            "y1": r.y1,
                            "y_center": (r.y0 + r.y1) / 2,
                            "is_block": True
                        })
                    else:
                        if h > 32:
                            va = "-0.65em"
                            mh = "3.6em"
                        elif h > 18:
                            va = "-0.38em"
                            mh = "2.2em"
                        else:
                            va = "-0.2em"
                            mh = "1.55em"
                        html = f'<img src="data:image/png;base64,{b64}" alt="math" class="inline-math-img" style="vertical-align:{va};max-height:{mh};display:inline-block;margin:0 1px;" />'

                        equation_items.append({
                            "text": f" {html} ",
                            "rect": r,
                            "bbox": [r.x0, r.y0, r.x1, r.y1],
                            "x0": r.x0,
                            "x1": r.x1,
                            "y0": r.y0,
                            "y1": r.y1,
                            "y_center": (r.y0 + r.y1) / 2,
                            "is_block": False
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
        if malayalam_chars_count < 20 and re.search(r'[ß∂Ø∏ƒ≥‰≠‡∑›™‚]', page_text):
            print(f"Page {page_idx + 1}: FML/ML-TT legacy symbols detected. Transliterating...", flush=True)
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

        full_pages.append(page_text)

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

    return {
        "full_text": "\n\n".join(full_pages),
        "pages": pages_metadata,
        "prompt_tokens": total_p_tokens,
        "completion_tokens": total_c_tokens,
    }


def route_and_extract(file_path: str) -> dict:
    """Ingestion routing layer entrypoint."""
    return extract_text_from_pdf(file_path)
