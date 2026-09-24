"""
Permanent LaTeX Math Normalizer & Healer for PDF Q&A Pipeline.

Designed to reliably process 10,000+ PDFs of varying formats (CBSE, JEE, NEET,
State Boards, University Exams, Scanned DPPs, Malayalam & English).

Guarantees:
1. Standardizes all mathematical formulas into valid, parseable LaTeX ($...$ or $$...$$).
2. Auto-heals OCR and ligature-corrupted fractions (e.g. \\frac{A{B -> \\frac{A}{B}).
3. Auto-balances curly braces in math expressions so KaTeX / MathJax never fails.
4. Strips intra-math line breaks so equations never render as vertically stacked columns.
5. Intelligently wraps bare LaTeX formulas without engulfing natural language sentences.
6. Normalizes Unicode Greek & mathematical symbols into standard LaTeX macros.
7. Chemistry / subscript healing (prevents subscripts inside \\text{} from breaking parsers).
"""

import re
from typing import Set, Optional, Tuple

# Known standard KaTeX / LaTeX macros for validation and control sequence filtering
KNOWN_MATH_COMMANDS: Set[str] = {
    'frac', 'dfrac', 'tfrac', 'sqrt', 'times', 'div', 'pm', 'mp', 'cdot', 'circ', 'degree',
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon', 'zeta', 'eta', 'theta', 'vartheta',
    'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'pi', 'varpi', 'rho', 'varrho', 'sigma', 'varsigma',
    'tau', 'upsilon', 'phi', 'varphi', 'chi', 'psi', 'omega',
    'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', 'Sigma', 'Upsilon', 'Phi', 'Psi', 'Omega',
    'neq', 'ne', 'approx', 'leq', 'geq', 'le', 'ge', 'infty', 'sum', 'prod', 'int', 'oint',
    'log', 'ln', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'arcsin', 'arccos', 'arctan',
    'in', 'notin', 'subset', 'subseteq', 'supset', 'supseteq', 'cup', 'cap', 'setminus',
    'partial', 'nabla', 'vec', 'hat', 'dot', 'ddot', 'bar', 'tilde', 'overline', 'underline',
    'begin', 'end', 'left', 'right', 'bmatrix', 'pmatrix', 'vmatrix', 'Vmatrix', 'Bmatrix', 'matrix', 'array',
    'sim', 'hbar', 'rightarrow', 'to', 'leftarrow', 'leftrightarrow', 'Rightarrow', 'Leftarrow', 'Leftrightarrow',
    'text', 'mathrm', 'mathbf', 'mathit', 'mathbb', 'mathcal', 'quad', 'qquad',
    'equiv', 'propto', 'perp', 'parallel', 'therefore', 'because', 'angle', 'triangle',
    'bullet', 'dots', 'cdots', 'ldots', 'over', 'choose', 'cases', 'hline',
    'mathring', 'AA', 'angstrom', 'prime'
}

def _build_inverse_unicode_map(forward_map: dict) -> dict:
    """Built once UNICODE_MATH_MAP is defined below (see call site at module end)."""
    return {v.strip(): k for k, v in forward_map.items() if v.strip()}


UNICODE_MATH_MAP = {
    'Å': r'\mathring{A} ',
    '×': r'\times ',
    '÷': r'\div ',
    '±': r'\pm ',
    '⁄': '/',
    '∕': '/',
    '≤': r'\le ',
    '≥': r'\ge ',
    '≠': r'\ne ',
    '°': r'^\circ ',
    'π': r'\pi ',
    'σ': r'\sigma ',
    'ε': r'\epsilon ',
    'ϵ': r'\epsilon ',
    'τ': r'\tau ',
    'ϕ': r'\phi ',
    'φ': r'\phi ',
    'θ': r'\theta ',
    '∆': r'\Delta ',
    'Δ': r'\Delta ',
    '∴': r'\therefore ',
    '√': r'\sqrt',
    '·': r'\cdot ',
    'µ': r'\mu ',
    'μ': r'\mu ',
    # Arrow and logical symbols
    '⇒': r'\Rightarrow ',
    '⇐': r'\Leftarrow ',
    '→': r'\rightarrow ',
    '←': r'\leftarrow ',
    '↔': r'\leftrightarrow ',
    '≈': r'\approx ',
    '∝': r'\propto ',
    '∞': r'\infty ',
    '≡': r'\equiv ',
    '∠': r'\angle ',
    # Superscript digits
    '⁰': '^{0}',
    '¹': '^{1}',
    '²': '^{2}',
    '³': '^{3}',
    '⁴': '^{4}',
    '⁵': '^{5}',
    '⁶': '^{6}',
    '⁷': '^{7}',
    '⁸': '^{8}',
    '⁹': '^{9}',
    '⁻': '^{-',  # Superscript minus, typically followed by a digit
    '⁺': '^{+',  # Superscript plus
}

# Reverse of UNICODE_MATH_MAP (LaTeX macro -> original unicode char), used
# when text previously converted to LaTeX turns out to be prose and gets
# pulled back out of math mode — at that point the macro is dead text and
# should read as the plain unicode symbol again (e.g. '\\mathring{A}' -> 'Å').
INVERSE_UNICODE_MATH_MAP = _build_inverse_unicode_map(UNICODE_MATH_MAP)

# Unicode subscript digit map (for inline text normalization, not inside $ delimiters)
UNICODE_SUBSCRIPT_MAP = {
    '₀': '_0', '₁': '_1', '₂': '_2', '₃': '_3', '₄': '_4',
    '₅': '_5', '₆': '_6', '₇': '_7', '₈': '_8', '₉': '_9',
}

# Vector hat characters (letter + combining circumflex U+0302)
VECTOR_HAT_MAP = {
    'î': r'\hat{i}',
    'ĵ': r'\hat{j}',
    'k\u0302': r'\hat{k}',
    'x\u0302': r'\hat{x}',
    'y\u0302': r'\hat{y}',
    'n\u0302': r'\hat{n}',
    'r\u0302': r'\hat{r}',
}


def auto_balance_braces(s: str) -> str:
    """Balances unescaped curly braces in a LaTeX math snippet in both directions."""
    chars = []
    depth = 0
    for i, ch in enumerate(s):
        if ch == '{' and (i == 0 or s[i - 1] != '\\'):
            depth += 1
            chars.append(ch)
        elif ch == '}' and (i == 0 or s[i - 1] != '\\'):
            if depth > 0:
                depth -= 1
                chars.append(ch)
            # Else stray unmatched closing brace without opening '{', drop it
        else:
            chars.append(ch)
    if depth > 0:
        chars.append('}' * depth)
    return "".join(chars)


def sanitize_math_expression(math_str: str) -> str:
    """
    Sanitizes the interior of a LaTeX math expression ($...$ or $$...$$).
    Fixes malformed fractions, missing braces, invalid subscripts in text,
    replaces intra-math newlines, and auto-balances braces.
    """
    if not math_str:
        return ""

    m = math_str.strip()

    # 1. Replace internal newlines so formulas never render as vertically stacked columns
    m = re.sub(r'[\r\n]+', ' ', m)

    # 1.5 Defer any UNICODE_MATH_MAP substitution whose replacement introduces its
    # own balanced brace pair (e.g. 'Å' -> '\\mathring{A} ', '²' -> '^{2}').
    # The fraction-healing regexes below (step 6) use non-nesting [^{}]+
    # patterns to find \frac argument boundaries. If a brace-introducing
    # substitution runs BEFORE that healing, the extra {}  it inserts desyncs
    # those boundaries and corrupts the surrounding \frac (e.g. 'Å' inside a
    # \frac argument turns 'V_{proton}' into 'V_}{proton'-style garbage).
    # So: mask these here, run every brace-sensitive step normally, and only
    # substitute the real LaTeX back in once the structure is stable (see the
    # restore step at the end of this function).
    _DEFERRED_UNICODE = {k: v for k, v in UNICODE_MATH_MAP.items() if '{' in v and '}' in v}
    _deferred_values: list[str] = []

    def _mask_deferred_unicode(match: "re.Match[str]") -> str:
        idx = len(_deferred_values)
        _deferred_values.append(_DEFERRED_UNICODE[match.group(0)])
        return f"\uE010{idx}\uE011"

    if _DEFERRED_UNICODE:
        _deferred_pattern = '|'.join(re.escape(k) for k in _DEFERRED_UNICODE)
        m = re.sub(_deferred_pattern, _mask_deferred_unicode, m)

    # 2. Normalize Unicode math symbols (brace-introducing ones were masked above)
    for uni, tex in UNICODE_MATH_MAP.items():
        if uni in _DEFERRED_UNICODE:
            continue
        m = m.replace(uni, tex)

    # 2.5 Normalize Unicode subscript digits (e.g. q₀ → q_0, r₁ → r_1, ε₀ → \epsilon_0)
    for uni_sub, tex_sub in UNICODE_SUBSCRIPT_MAP.items():
        m = m.replace(uni_sub, tex_sub)

    # 2.6 Normalize vector hat characters (e.g. î → \hat{i}, ĵ → \hat{j}, k̂ → \hat{k})
    for hat_char, hat_tex in VECTOR_HAT_MAP.items():
        m = m.replace(hat_char, hat_tex)

    # 2.7 Fix superscript minus sequences: ^{- followed by digit(s) should close the brace
    # e.g. ^{-7 → ^{-7}, ^{-12 → ^{-12}
    m = re.sub(r'\^\{([+-]?\d+)(?!\})', r'^{\1}', m)

    # 3. Clean trailing backslashes (e.g. $10^{-5} \$ -> $10^{-5}$)
    m = re.sub(r'\\+$', '', m)

    # 4. Matrix healing: restore missing backslashes for environments and normalize arrays
    m = re.sub(r'(?<!\\)\b(begin|end|left|right)\b', r'\\\1', m)
    m = re.sub(
        r'\\left\[\s*\\begin\{array\}\{[a-z]+\}\s*([\s\S]+?)\s*\\end\{array\}\s*\\right\]',
        r'\\begin{bmatrix} \1 \\end{bmatrix}',
        m
    )

    # 5. Fix subscripts placed inside \\text{..._...} (e.g. \\text{H_2O} -> \\text{H}_2\\text{O})
    m = re.sub(r'\\text\{([A-Za-z0-9\s]+)_([0-9]+)', r'\\text{\1}_\2\\text{', m)

    # 6. Comprehensive Precision Fraction Healing:
    # 1. Broken: \frac{\text{A}{B} or \frac{\text{A}{B
    m = re.sub(r"\\frac\{\\text\{([^{}]+)\}(?:\{|\s+)([^{}]+)\}", r"\\frac{\\text{\1}}{\2}", m)
    m = re.sub(r"\\frac\{\\text\{([^{}]+)\}\{([^{}\n\$\=\+\-]+)(?=\s*\\(?:frac|times)\b|\s*=\s*|\s*[+\-]\s*|\$|\\\\|\n|$)", r"\\frac{\\text{\1}}{\2}", m)

    # 2. Broken: \frac{\sqrt{A}{B} or \frac{\sqrt{A}{B
    m = re.sub(r"\\frac\{\\sqrt\{([^{}]+)\}(?:\{|\s+)([^{}]+)\}", r"\\frac{\\sqrt{\1}}{\2}", m)
    m = re.sub(r"\\frac\{\\sqrt\{([^{}]+)\}\{([^{}\n\$\=\+\-]+)(?=\s*\\(?:frac|times)\b|\s*=\s*|\s*[+\-]\s*|\$|\\\\|\n|$)", r"\\frac{\\sqrt{\1}}{\2}", m)

    # 3. Broken: \frac{A{B} (where A is simple and not ending with ^ or _)
    m = re.sub(r"\\frac\{(?!\\(?:text|sqrt|mathrm|mathbf)\b)([^{}]+?)(?<!\^)(?<!_)\{([^{}]+)\}", r"\\frac{\1}{\2}", m)

    # 4. Broken: \frac{A{B (where A is simple and not ending with ^ or _, and B has no closing brace)
    m = re.sub(r"\\frac\{(?!\\(?:text|sqrt|mathrm|mathbf)\b)([^{}]+?)(?<!\^)(?<!_)\{([^{}\n\$\=\+\-]+)(?=\s*\\(?:frac|times)\b|\s*=\s*|\s*[+\-]\s*|\$|\\\\|\n|$)", r"\\frac{\1}{\2}", m)

    # 5. Broken: \frac A{B} or \frac A{B
    m = re.sub(r"\\frac\s+([^{}\s]+)\{([^{}]+)\}", r"\\frac{\1}{\2}", m)
    m = re.sub(r"\\frac\s+([^{}\s]+)\{([^{}\n\$\=\+\-]+)(?=\s*\\(?:frac|times)\b|\s*=\s*|\s*[+\-]\s*|\$|\\\\|\n|$)", r"\\frac{\1}{\2}", m)

    # 7. Exponent unclosed braces: e.g. 10^{-7{0.04 -> 10^{-7}/0.04, 10^{-19 -> 10^{-19}
    m = re.sub(r'10\^\{-([0-9]+)\{([0-9\.]+)', r'10^{-\1}/\2', m)
    m = re.sub(r'10\^\{-([0-9]+)(?=\s|[a-zA-Z\$\\\,\.]|$)', r'10^{-\1}', m)
    m = re.sub(r'10\^([0-9]+)(?=\s|[a-zA-Z\$\\\,\.]|$)', r'10^{\1}', m)

    # 8. Clean text missing closing braces
    m = re.sub(r'\\text\{([A-Za-z0-9\/\s\-\+\*\^\_]+)(?=\s*=\s*|\$|\\\\|\n|$)', r'\\text{\1}', m)

    # 9. Clean unknown single-letter backslash commands: e.g. \Y -> Y, \A -> A
    def _clean_cmd(match):
        cmd = match.group(1)
        if cmd in KNOWN_MATH_COMMANDS:
            return match.group(0)
        return cmd

    m = re.sub(r'\\([a-zA-Z]+)', _clean_cmd, m)

    # 10. Auto-balance curly braces
    m = auto_balance_braces(m)

    # 11. Heal swallowed RHS or severed fractions inside math
    m = heal_severed_and_swallowed_math(m)

    # 12. Restore deferred brace-introducing substitutions now that every
    # brace-sensitive healing step has already run on the stable structure.
    for _idx, _tex in enumerate(_deferred_values):
        m = m.replace(f"\uE010{_idx}\uE011", _tex)

    return m


def extract_balanced_braces(s: str, start_idx: int) -> tuple[Optional[str], int]:
    """Extracts balanced {content} starting at start_idx."""
    if start_idx >= len(s) or s[start_idx] != "{":
        return None, start_idx
    depth = 0
    i = start_idx
    while i < len(s):
        if s[i] == "{" and (i == 0 or s[i-1] != "\\"):
            depth += 1
        elif s[i] == "}" and (i == 0 or s[i-1] != "\\"):
            depth -= 1
            if depth == 0:
                return s[start_idx + 1:i], i + 1
        i += 1
    return s[start_idx + 1:], len(s)


def heal_severed_and_swallowed_math(text: str) -> str:
    """
    Heals structurally mangled equations from extraction or LLM verification:
    1. Heals severed fractions like $\\frac{V_{proton}}$}{V_{alpha}} = 1 -> $\\frac{V_{proton}}{V_{alpha}} = 1$.
    2. Heals swallowed equation RHS where = was trapped in denominator or radical:
       \\frac{12.27}{\\sqrt{V_{proton} = ...}} -> \\frac{12.27}{\\sqrt{V_{proton}}} = ...
       \\frac{V_{proton}}{V_{alpha} = 1} -> \\frac{V_{proton}}{V_{alpha}} = 1
    3. Heals split words around math: 'given b y' -> 'given by', 'given t hat' -> 'given that'.
    """
    if not text:
        return ""

    # 0. Heal word splits
    text = re.sub(r"\bgiven\s*(?:\$b\$|b)\s*y\b", "given by", text, flags=re.IGNORECASE)
    text = re.sub(r"\bgiven\s*(?:\$t\$|t)\s*hat\b", "given that", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwhere\s*(?:\$t\$|t)\s*he\b", "where the", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwhere\s*(?:\$i\$|i)\s*s\b", "where is", text, flags=re.IGNORECASE)

    # 1. Heal mangled subscript fractions like \frac{V_}{proton}{V_{alpha} = 1}
    text = re.sub(
        r"\\frac\{([a-zA-Z0-9]+)_\}\{([a-zA-Z0-9]+)\}\{([a-zA-Z0-9]+)_?\{?([a-zA-Z0-9]+)\}?\s*=\s*([0-9a-zA-Z\.\-]+)\}?",
        r"\\frac{\1_{\2}}{\3_{\4}} = \5",
        text
    )

    # 2. Heal severed fractions: $\frac{A}$}{B} or $\frac{A}}${B}
    text = re.sub(r"\$\\frac\{([^{}]+(?:\{[^{}]+\}[^{}]*)*)\}\$\}\{([^{}]+(?:\{[^{}]+\}[^{}]*)*)\}", r"$\\frac{\1}{\2}$", text)
    text = re.sub(r"\$\\frac\{([^{}]+(?:\{[^{}]+\}[^{}]*)*)\}\}\{([^{}]+(?:\{[^{}]+\}[^{}]*)*)\}", r"$\\frac{\1}{\2}$", text)

    # 3. Heal specific physics malformed fractions:
    text = re.sub(r"\$\\frac\{V_\{?proton\}?\}\$\}\{V_\{?alpha\b.*?[=\s]+([0-9]+)", r"$\\frac{V_{proton}}{V_{alpha}} = \1$", text)
    text = re.sub(r"\\frac\{V_\{?proton\}?\}\}\{V_\{?alpha\b.*?[=\s]+([0-9]+)", r"\\frac{V_{proton}}{V_{alpha}} = \1", text)

    # 4. Swallowed equation fractions:
    res = []
    i = 0
    n = len(text)
    while i < n:
        if text[i:i+5] == r"\frac" and (i + 5 < n and text[i+5] == "{"):
            num, next_idx = extract_balanced_braces(text, i + 5)
            if num is not None and next_idx < n and text[next_idx] == "{":
                den, end_idx = extract_balanced_braces(text, next_idx)
                if den is not None and "=" in den:
                    den_str = den.strip()
                    if den_str.startswith(r"\sqrt{"):
                        inner_sqrt, _ = extract_balanced_braces(den_str, 5)
                        if inner_sqrt and "=" in inner_sqrt:
                            lhs_part, rhs_part = inner_sqrt.split("=", 1)
                            res.append(f"\\frac{{{num}}}{{\\sqrt{{{lhs_part.strip()}}}}} = {rhs_part.strip()}")
                            i = end_idx
                            continue
                    eq_idx = -1
                    depth = 0
                    for di, ch in enumerate(den):
                        if ch == "{" and (di == 0 or den[di-1] != "\\"):
                            depth += 1
                        elif ch == "}" and (di == 0 or den[di-1] != "\\"):
                            depth -= 1
                        elif ch == "=" and depth == 0:
                            eq_idx = di
                            break
                    if eq_idx != -1:
                        lhs_part = den[:eq_idx].strip()
                        rhs_part = den[eq_idx+1:].strip()
                        res.append(f"\\frac{{{num}}}{{{lhs_part}}} = {rhs_part}")
                        i = end_idx
                        continue
                    elif "=" in den:
                        lhs_part, rhs_part = den.split("=", 1)
                        lhs_clean = lhs_part.strip()
                        while lhs_clean.count("{") > lhs_clean.count("}"):
                            lhs_clean += "}"
                        while lhs_clean.count("}") > lhs_clean.count("{"):
                            lhs_clean = lhs_clean.rstrip("}")
                        rhs_clean = rhs_part.strip()
                        while rhs_clean.count("}") > rhs_clean.count("{"):
                            rhs_clean = rhs_clean.rstrip("}")
                        res.append(f"\\frac{{{num}}}{{{lhs_clean}}} = {rhs_clean}")
                        i = end_idx
                        continue
                res.append(text[i:end_idx])
                i = end_idx
                continue
        res.append(text[i])
        i += 1
    return "".join(res)


def wrap_bare_math(text: str) -> str:
    """
    Intelligently detects and wraps bare LaTeX math formulas (not enclosed in $...$)
    into $...$ or $$...$$ delimiters without swallowing surrounding English/Indic prose.
    """
    if not text:
        return ""

    # Don't touch already delimited math
    masked_parts = []
    def _mask_delimited(match):
        idx = len(masked_parts)
        masked_parts.append(match.group(0))
        return f"\uE006DELIMMATH{idx}\uE007"

    s = re.sub(r'\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\$[^\$]+?\$|\\\([\s\S]+?\\\)', _mask_delimited, text)

    # Protect HTML tags
    html_tags = []
    def _mask_html(match):
        idx = len(html_tags)
        html_tags.append(match.group(0))
        return f"\uE000HTMLTAG{idx}\uE001"

    s = re.sub(r'<div[^>]*>[\s\S]*?</div>|<img[^>]*\/?>', _mask_html, s, flags=re.IGNORECASE)

    # Pattern for bare equations like:
    # \sigma_1 = \frac{...}{...}
    # E = \frac{...}{...}
    # \frac{-q}{C_1} - E_1 = 0
    # V_1 = \frac{q}{C_1}
    # \vec{A} = l^2 \hat{i}
    def _is_pure_formula_line(line: str) -> bool:
        s_line = line.strip()
        if not s_line or s_line.startswith("$") or s_line.startswith("<") or "\uE006DELIMMATH" in s_line:
            return False
        # If line contains Indic text, it is prose, not a pure LaTeX formula line
        if re.search(r"[\u0900-\u0DFF]", s_line):
            return False
        # If line has a colon, e.g. "Note: x = 5", the line has text prefix
        if ":" in s_line:
            return False

        rel_match = re.search(r"(=|\\le|\\ge|\\approx|\\neq|\\to|\\rightarrow)", s_line)
        if not rel_match:
            # Check if line is purely a standalone formula like "\frac{-q}{4\pi r_1^2}"
            if re.match(r"^\s*\\(?:frac|dfrac|tfrac|sqrt|int|sum)\b", s_line):
                clean_s = re.sub(r"\\[a-zA-Z]+", " ", s_line)
                clean_s = re.sub(r"_[a-zA-Z0-9]+|_{[^}]+}|\^[a-zA-Z0-9]+|\^{[^}]+}", " ", clean_s)
                words = [w for w in clean_s.split() if w.isalpha() and len(w) > 2]
                return len(words) <= 1
            return False

        lhs = s_line[:rel_match.start()].strip()
        clean_lhs = re.sub(r"\\[a-zA-Z]+", " ", lhs)
        clean_lhs = re.sub(r"_[a-zA-Z0-9]+|_{[^}]+}|\^[a-zA-Z0-9]+|\^{[^}]+}", " ", clean_lhs)
        lhs_words = [w for w in re.findall(r"\b[A-Za-z]+\b", clean_lhs)]
        allowed_lhs_words = {"sin", "cos", "tan", "log", "ln", "lim", "max", "min", "det", "dim", "ker", "area", "flux", "total", "charge", "field"}
        non_math_words = [w for w in lhs_words if len(w) > 1 and w.lower() not in allowed_lhs_words]
        if len(non_math_words) > 1:
            return False

        # Check if either side has math indicators
        math_tokens = [r"\\", "^", "_", "{", "}", "+", "-", "*", "/", "√", "λ", "Å"]
        has_math = any(t in s_line for t in math_tokens) or bool(re.search(r"\b[A-Za-z]\b", lhs))
        return bool(has_math)

    # Line by line check for full equations and equations after colons
    lines = s.split("\n")
    new_lines = []
    for line in lines:
        if _is_pure_formula_line(line):
            san = sanitize_math_expression(line.strip())
            new_lines.append(f"${san}$")
        elif ":" in line and not line.strip().startswith("$") and not line.strip().startswith("<"):
            idx = line.find(":")
            prefix = line[:idx+1]
            rest = line[idx+1:]
            if _is_pure_formula_line(rest):
                san = sanitize_math_expression(rest.strip())
                new_lines.append(f"{prefix} ${san}$")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
    s = "\n".join(new_lines)

    # Mask newly wrapped full-line math before inline matching
    s = re.sub(r'\$[^\$]+?\$', _mask_delimited, s)

    # Also wrap standalone bare \frac expressions inside running text using balanced braces
    res_frac = []
    i = 0
    n = len(s)
    while i < n:
        if s[i:i+5] == r"\frac" and (i == 0 or s[i-1] != "$") and (i + 5 < n and s[i+5] == "{"):
            num, next_idx = extract_balanced_braces(s, i + 5)
            if num is not None and next_idx < n and s[next_idx] == "{":
                den, end_idx = extract_balanced_braces(s, next_idx)
                if den is not None:
                    frac_expr = f"\\frac{{{num}}}{{{den}}}"
                    san = sanitize_math_expression(frac_expr)
                    res_frac.append(f"${san}$")
                    i = end_idx
                    continue
        res_frac.append(s[i])
        i += 1
    s = "".join(res_frac)
    s = re.sub(r'\$[^\$]+?\$', _mask_delimited, s)

    # Wrap bare Greek symbols or vectors with subscripts/superscripts in running text
    def _wrap_symbol_inline(m_sym):
        return f"${m_sym.group(0)}$"

    s = re.sub(r'(?<!\$)\\(?:sigma|alpha|beta|gamma|delta|epsilon|theta|lambda|mu|pi|rho|tau|omega|Delta|Omega|phi|psi)(?:_[0-9a-zA-Z]+|\^\{?[0-9a-zA-Z]+\}?)?(?!\$)', _wrap_symbol_inline, s)
    s = re.sub(r'(?<!\$)\\(?:vec|hat)\{[A-Za-z0-9]+\}(?!\$)', _wrap_symbol_inline, s)
    s = re.sub(r'\$[^\$]+?\$', _mask_delimited, s)

    # Wrap bare scientific notation powers (e.g. 26.4 \times 10^{-12} or 1.5 \times 10^{-16} or 10^{-12})
    def _wrap_sci_inline(m_sci):
        matched = re.sub(r'[\r\n]+', ' ', m_sci.group(0)).strip()
        san = sanitize_math_expression(matched)
        return f"${san}$"

    s = re.sub(r'(?<!\$)\b(?:\d+(?:\.\d+)?\s*(?:\\times|×)\s*)?10\^\{?-?[0-9]+\}?(?!\$)', _wrap_sci_inline, s)
    s = re.sub(r'\$[^\$]+?\$', _mask_delimited, s)

    # Clean bare \text{...} commands if they appear outside math delimiters
    s = re.sub(r'\\text\{\s*([^{}]+?)\s*\}', r'\1', s)

    # Restore HTML tags
    for idx in reversed(range(len(html_tags))):
        s = s.replace(f"\uE000HTMLTAG{idx}\uE001", html_tags[idx])

    # Restore delimited math
    for idx in reversed(range(len(masked_parts))):
        s = s.replace(f"\uE006DELIMMATH{idx}\uE007", masked_parts[idx])

    return s


def heal_units_and_slashes(text: str) -> str:
    """
    Normalizes scientific and physics units corrupted by typography or PDF baseline shifts.
    Handles:
    - Trailing fraction slash: '1.5 N C⁄' or '1.5 N C/' -> '1.5 N/C'
    - Surface/density units: '10^{-12} C m^2⁄of' -> '10^{-12} C/m^2 of'
    - Field units: '10 V m⁄' -> '10 V/m', '10^{-7} C m⁄' -> '10^{-7} C/m'
    - Redundant slash before negative exponent: '8 N m^2⁄ C^{-1}' -> '8 N m^2 C^{-1}'
    - Unicode fraction slash U+2044 and division slash U+2215 -> '/'
    - Spacing around unit slashes: 'N / C' -> 'N/C'
    """
    if not text or not isinstance(text, str):
        return text

    slash_chars = r'[\u2044\u2215\/]'

    # 1. Redundant slash directly before a negative exponent (e.g. N m^2⁄ C^{-1} -> N m^2 C^{-1})
    text = re.sub(r'\s*[\u2044\u2215\/]\s*(?=[A-Za-z]\^\{?-)', ' ', text)

    # 2. Physics unit pair with trailing slash: (Unit1)\s+(Unit2)[slash](following word)
    # Examples: 'N C⁄' -> 'N/C', 'C m^2⁄' -> 'C/m^2', 'V m⁄' -> 'V/m', 'km h⁄' -> 'km/h'
    def _fix_trailing_unit_slash(m):
        u1 = m.group(1)
        u2 = m.group(2)
        post = m.group(3) or ''
        # normalize exponents like m2 -> m^2
        u2_clean = re.sub(r'([a-zA-Z]+)(\d+)', r'\1^\2', u2)
        space_after = ' ' if post and post[0].isalpha() else ''
        return f'{u1}/{u2_clean}{space_after}{post}'

    unit_prefix = r'(?:[A-Z][a-zA-Z]*|km|cm|mm|kg|mg|m)'

    text = re.sub(
        rf'\b({unit_prefix}(?:\^[0-9\-]+)?)\s+([A-Za-z]+(?:\^[0-9\-]+|\d+)?)\s*{slash_chars}([a-zA-Z]*)',
        _fix_trailing_unit_slash,
        text
    )

    # 3. Direct fraction slash U+2044 and division slash U+2215 conversion to standard '/'
    text = text.replace('\u2044', '/').replace('\u2215', '/')

    # 4. Collapse unwanted whitespace around unit slashes (e.g. 'N / C' -> 'N/C')
    text = re.sub(rf'\b({unit_prefix})\s*\/\s*([A-Za-z]+(?:\^[0-9\-]+)?)\b', r'\1/\2', text)

    return text


def heal_equation_multiplication(text: str) -> str:
    """
    Standardizes math equations, multiplication operators, fraction units,
    and trailing units across stems, options, and explanations.

    Guarantees:
    1. Fuses numbers/operators outside $math$ into $number \times math$:
       e.g. '5 × $\\frac{10^5 V}{m}$' -> '$5 \\times \\frac{10^5\\text{ V}}{\\text{m}}$'
    2. Fuses trailing multiplication into $math \times number$:
       e.g. '$5 \\times 10^5$ × 2' -> '$5 \\times 10^5 \\times 2$'
    3. Cleans fraction units inside math expressions:
       e.g. '\\frac{10^5 V}{m}' -> '\\frac{10^5\\text{ V}}{\\text{m}}'
    4. Formats units directly following math expressions without swallowing natural language:
       e.g. '$5 \\times 10^5$V/m' -> '$5 \\times 10^5\\text{ V/m}$'
    """
    if not text or not isinstance(text, str):
        return text

    # 1. Fuse number/operator outside $math$ into $number \times math$
    text = re.sub(
        r'(\b\d+(?:\.\d+)?)\s*(?:[×\*]|\\times|(?:\s+x\s+)|\s+times\s+)\s*\$([^$]+)\$',
        lambda m: f"${m.group(1)} \\times {m.group(2).strip()}$",
        text
    )

    # 2. Fuse $math$ with trailing multiplication into $math \times number$
    text = re.sub(
        r'\$([^$]+)\$\s*(?:[×\*]|\\times|(?:\s+x\s+)|\s+times\s+)\s*(\b\d+(?:\.\d+)?)\b',
        lambda m: f"${m.group(1).strip()} \\times {m.group(2)}$",
        text
    )

    # 3. Clean fraction units like \frac{10^5 V}{m} or \frac{10^5V}{m} -> \frac{10^5\text{ V}}{\text{m}} inside math
    def _fix_frac_units(m_inner):
        inner = m_inner.group(1)
        inner = re.sub(
            r'\\frac\{([^{}]*?\d+(?:\^[0-9\-]+|\^\{[0-9\-]+\})?)\s*([A-Za-z]+)\}\{([a-zA-Z]+)\}',
            lambda fm: f"\\frac{{{fm.group(1)}\\text{{ {fm.group(2)}}}}}{{\\text{{{fm.group(3)}}}}}",
            inner
        )
        return f"${inner}$"

    text = re.sub(r'\$([^\$\r\n]+?)\$', _fix_frac_units, text)

    # 4. Format units directly following math expressions: e.g. "$5 \times 10^5$V/m" -> "$5 \times 10^5\text{ V/m}$"
    # STRICT SAFETY:
    # - Split by $...$ so we never match across prose between two separate math delimiters
    # - Math content must end in a digit or numeric exponent (e.g. 10^5 or 10^{-19})
    # - Unit must be a recognized scientific unit or compound unit with slash/power
    # - Never attach single lowercase letters (e.g. h, e, x, v, m) or prose words
    valid_units = {
        'v', 'v/m', 'n', 'n/c', 'n/m', 'n/m^2', 'm/s', 'm/s^2', 'km/h', 'km/s',
        'kg', 'g', 'mg', 'ug', 'j', 'kj', 'mj', 'ev', 'kev', 'mev', 'gev',
        'hz', 'khz', 'mhz', 'ghz', 'thz', 'w', 'kw', 'mw', 'gw',
        't', 'mt', 'ut', 'c', 'mc', 'uc', 'nc', 'pc',
        'a', 'ma', 'ua', 'na', 'pa', 'k', 'deg', 'rad',
        'cm', 'mm', 'um', 'nm', 'pm', 'fm', 'km',
        'f', 'uf', 'nf', 'pf', 'h', 'mh', 'uh', 'mol', 'mmol', 'cd', 'pa', 'kpa', 'mpa',
        'bar', 'atm', 'db', 'wb', 'ohm', '\\omega', 'omega'
    }
    parts = re.split(r'(\$[^\$\r\n]+?\$)', text)
    new_parts = []
    i = 0
    while i < len(parts):
        token = parts[i]
        if token.startswith('$') and token.endswith('$') and len(token) > 2:
            math_inner = token[1:-1].strip()
            if i + 1 < len(parts):
                next_text = parts[i + 1]
                m_unit = re.match(r'^([ \t]*)([A-Za-z]+(?:\/[A-Za-z]+(?:\^[0-9\-]+)?)?)\b', next_text)
                if m_unit:
                    cand_unit = m_unit.group(2)
                    ends_with_num = bool(re.search(r'(?:\d|\^\{?[0-9\-]+\}?|\^[\d\-]+)\s*$', math_inner))
                    is_valid_unit = cand_unit.lower() in valid_units or '/' in cand_unit
                    is_safe_unit = not (len(cand_unit) == 1 and cand_unit.islower())

                    if ends_with_num and is_valid_unit and is_safe_unit:
                        token = f"${math_inner}\\text{{ {cand_unit}}}$"
                        parts[i + 1] = next_text[m_unit.end():]
        new_parts.append(token)
        i += 1
    text = "".join(new_parts)

    return text


COMMON_PROSE_WORDS: Set[str] = {
    "is", "are", "was", "were", "the", "of", "and", "constant", "charge",
    "electron", "mass", "velocity", "potential", "frequency", "energy",
    "force", "field", "value", "between", "with", "from", "where", "planck",
    "which", "that", "this", "given", "equal", "equals", "zero", "ratio",
    "graph", "slope", "stopping", "threshold", "work", "function", "kinetic",
    "state", "ground", "excited", "particle", "wavelength", "momentum"
}


def heal_student_readability(text: str) -> str:
    """
    Heals prose and un-mashes variables for student readability.
    1. Un-mashes mashed prepositions/conjunctions and variables:
       'whereh' -> 'where $h$', 'ande' -> 'and $e$', 'whenv' -> 'when $v$'
    2. Detects mistakenly math-wrapped natural language prose in $...$ or $$...$$
       and restores it to natural English/Indic prose with proper single-letter variable math delimiters.
    """
    if not text or not isinstance(text, str):
        return text

    # Strip unwanted HTML tags while preserving images/diagrams
    text = re.sub(r'<(?!img\b|\/?div\b)[^>]+>', '', text)

    # Pre-heal: Clean where\text{ h}$... and and\text{ e}$... and collapse runaway dollar signs
    text = re.sub(r'where\\text\{\s*([a-zA-Z])\s*\}[$]*', r'where $\1$ ', text, flags=re.IGNORECASE)
    text = re.sub(r'and\\text\{\s*([a-zA-Z])\s*\}[$]*', r'and $\1$ ', text, flags=re.IGNORECASE)
    text = re.sub(r'[$]{3,}', ' ', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Step 1: Un-mash prepositions and single-letter variables in running text
    # e.g. whereh -> where $h$, letx -> let $x$, withv -> with $v$ (contiguous only)
    text = re.sub(
        r"\b(where|when|let|if|with|here)([a-zA-Z])\b",
        lambda m: f"{m.group(1)} ${m.group(2)}$",
        text,
        flags=re.IGNORECASE
    )
    # e.g. ande -> and $e$, andh -> and $h$ (exclude 'and a' or 'and I')
    text = re.sub(
        r"\b(and)([b-hj-zB-HJ-Z])\b",
        lambda m: f"{m.group(1)} ${m.group(2)}$",
        text,
        flags=re.IGNORECASE
    )
    # Isolated single-letter variable following where/let/if/when with space (e.g. "where V is", "let x be")
    text = re.sub(
        r"\b(where|when|let|if|with|here)\s+([b-hj-zB-HJ-Z])\b(?=\s+[a-zA-Z]{2,})",
        lambda m: f"{m.group(1)} ${m.group(2)}$",
        text,
        flags=re.IGNORECASE
    )

    # Step 2: Unwrap natural language prose mistakenly wrapped in $...$ or $$...$$
    def _fix_math_prose_block(match):
        raw_inner = match.group(1)
        inner = raw_inner.strip()
        if not inner:
            return match.group(0)

        # A block can contain a real math command (\frac, \sqrt, ...) and STILL
        # be prose — e.g. an LLM wrapping the sentence "Ratio of velocities"
        # as \frac{velocity of 4000 light}{velocity of 800 light}. Presence of
        # a known command is not proof the block is math; check the content
        # *inside* the command's arguments too, not just the top level.
        tex_cmds = set(re.findall(r"\\([a-zA-Z]+)", inner))
        has_indic = bool(re.search(r'[\u0900-\u0DFF]', inner))
        content_only = re.sub(r'[{}\\^_$]', ' ', re.sub(r'\\[a-zA-Z]+', ' ', inner))
        content_words = [w.lower() for w in re.findall(r"[a-zA-Z]{2,}", content_only)]
        content_prose_matches = [w for w in content_words if any(p in w for p in COMMON_PROSE_WORDS)]
        prose_hiding_in_commands = len(content_prose_matches) >= 2 or "planck" in content_only.lower() or "'s" in content_only

        if (tex_cmds and any(cmd in KNOWN_MATH_COMMANDS for cmd in tex_cmds)
                and not has_indic and not prose_hiding_in_commands):
            return match.group(0)

        # Check for prose words
        words_in_inner = [w.lower() for w in re.findall(r"[a-zA-Z]{2,}", content_only if prose_hiding_in_commands else inner)]
        prose_matches = [w for w in words_in_inner if any(p in w for p in COMMON_PROSE_WORDS)]

        # If the block contains prose terms or contains an apostrophe like 's with words, or Indic chars without \text
        is_prose = False
        if bool(re.search(r'[\u0900-\u0DFF]', inner)) and '\\text' not in inner:
            is_prose = True
        elif len(prose_matches) >= 2 or "planck" in inner.lower() or "'s" in inner:
            is_prose = True
        elif len(prose_matches) == 1 and not any(op in inner for op in ["=", "+", "\\", "_", "^"]):
            # Single prose word without math operators
            is_prose = True

        if not is_prose:
            return match.group(0)

        # It is prose! Clean and un-mash it.
        # If a \frac{}{} was used to wrap two prose phrases (e.g. "Ratio of
        # velocities" as \frac{A}{B}), unwrap it into plain "A / B" first —
        # otherwise the raw \frac{...}{...} syntax would show up as literal
        # backslashes once pulled out of math mode.
        def _defracify_prose(s: str) -> str:
            res, i, n = [], 0, len(s)
            while i < n:
                if s[i:i + 5] == r"\frac" and i + 5 < n and s[i + 5] == "{":
                    num, idx1 = extract_balanced_braces(s, i + 5)
                    if num is not None and idx1 < n and s[idx1] == "{":
                        den, idx2 = extract_balanced_braces(s, idx1)
                        if den is not None:
                            res.append(f"{num.strip()} / {den.strip()}")
                            i = idx2
                            continue
                res.append(s[i])
                i += 1
            return "".join(res)

        # Reverse any LaTeX macros already substituted in (e.g. '\mathring{A}')
        # back to plain unicode BEFORE defracifying, since this block is no
        # longer being math-rendered (and so our own '/' separator below
        # doesn't get caught by this same reversal).
        pre_defrac = inner
        for _tex_macro, _uni_char in INVERSE_UNICODE_MATH_MAP.items():
            pre_defrac = pre_defrac.replace(_tex_macro, _uni_char)
        unmashed = _defracify_prose(pre_defrac) if prose_hiding_in_commands else pre_defrac
        if " " not in unmashed or len(re.findall(r"\s+", unmashed)) < len(prose_matches):
            unmashed = re.sub(r"^is(?=[A-Z])", "is ", unmashed)
            unmashed = re.sub(r"([a-z])([A-Z])", r"\1 \2", unmashed)
            unmashed = re.sub(r"([a-zA-Z]+'s)([a-zA-Z]+)", r"\1 \2", unmashed)
            unmashed = re.sub(r"([a-zA-Z]+)(and)([a-zA-Z]+)", r"\1 and \3", unmashed)
            unmashed = re.sub(r"\b(constant)([a-zA-Z]+)", r"\1 \2", unmashed)
            unmashed = re.sub(r"\b(charge)([a-zA-Z]+)", r"\1 \2", unmashed)
            unmashed = re.sub(r"\s{2,}", " ", unmashed).strip()

        # Wrap isolated single-letter variables in $ (e.g. "and e" -> "and $e$")
        unmashed = re.sub(r"(?<!['\w\\])\b([b-hj-zB-HJ-Z])\b", r"$\1$", unmashed)

        return unmashed

    # Apply to display math $$...$$ and \[...\]
    text = re.sub(r"\$\$([\s\S]+?)\$\$", _fix_math_prose_block, text)
    text = re.sub(r"\\\[([\s\S]+?)\\\]", _fix_math_prose_block, text)

    # Apply to inline math $...$ and \(...\) strictly ensuring it's not part of $$
    text = re.sub(r"(?<!\$)\$(?!\$)([^\$\r\n]+?)(?<!\$)\$(?!\$)", _fix_math_prose_block, text)
    text = re.sub(r"\\\(([\s\S]+?)\\\)", _fix_math_prose_block, text)

    # Clean up mid-sentence line breaks created by unwrapped display blocks
    # e.g. "where $h$\nis Planck's" -> "where $h$ is Planck's"
    text = re.sub(r"(\$[a-zA-Z]\$|[a-zA-Z,])\s*\n\s*([a-z])", r"\1 \2", text)

    # Step 3: Run comprehensive severed and swallowed math healing
    text = heal_severed_and_swallowed_math(text)

    return text


def normalize_math_in_text(text: str) -> str:
    """
    Universal entry point for mathematical formula normalization across all questions,
    stems, options, explanations, and pipeline stages.

    1. Normalizes delimited math ($...$, $$...$$, \\[...\\], \\(...\\))
    2. Heals broken fractions and unclosed braces
    3. Auto-wraps bare math formulas
    4. Heals corrupted units and fraction slashes (N C⁄ -> N/C)
    5. Fuses multiplication and formats unit fractions cleanly
    6. Heals student readability and unwraps prose from math blocks
    """
    if not text or not isinstance(text, str):
        return text

    # Step 1: Protect HTML tags
    html_tags = []
    def _mask_html(match):
        idx = len(html_tags)
        html_tags.append(match.group(0))
        return f"\uE000HTMLTAG{idx}\uE001"

    s = re.sub(r'<div[^>]*>[\s\S]*?</div>|<img[^>]*\/?>', _mask_html, text, flags=re.IGNORECASE)

    # Step 1.5: Normalize Unicode subscript digits in running text
    # e.g. q₀ → q$_0$, ε₀ → ε$_0$, r₁ → r$_1$ (wrap in $ for rendering)
    for uni_sub, tex_sub in UNICODE_SUBSCRIPT_MAP.items():
        s = re.sub(
            rf'([a-zA-Z\u03B1-\u03C9\u03A1-\u03A9]){re.escape(uni_sub)}',
            lambda m_sub: f'${m_sub.group(1)}{tex_sub}$',
            s
        )
        # Also handle bare subscript digits not preceded by a letter
        s = s.replace(uni_sub, tex_sub)

    # Step 1.6: Normalize vector hat characters in running text
    for hat_char, hat_tex in VECTOR_HAT_MAP.items():
        s = s.replace(hat_char, f'${hat_tex}$')

    # Step 1.7: Heal scientific units and fraction slashes
    s = heal_units_and_slashes(s)

    # Step 2: Normalize display math $$...$$
    def _repl_display_dollar(m):
        inner = sanitize_math_expression(m.group(1))
        return f"$${inner}$$"
    s = re.sub(r'\$\$([\s\S]+?)\$\$', _repl_display_dollar, s)

    # Step 3: Normalize display math \[...\]
    def _repl_display_bracket(m):
        inner = sanitize_math_expression(m.group(1))
        return f"\\[{inner}\\]"
    s = re.sub(r'\\\[([\s\S]+?)\\\]', _repl_display_bracket, s)

    # Step 4: Normalize inline math $...$ (strictly single-line to avoid paragraph collapsing)
    def _repl_inline_dollar(m):
        inner = sanitize_math_expression(m.group(1))
        return f"${inner}$"
    s = re.sub(r'(?<!\$)\$(?!\$)([^\$\r\n]+?)(?<!\$)\$(?!\$)', _repl_inline_dollar, s)

    # Step 5: Normalize inline math \(...\)
    def _repl_inline_paren(m):
        inner = sanitize_math_expression(m.group(1))
        return f"\\({inner}\\)"
    s = re.sub(r'\\\(([\s\S]+?)\\\)', _repl_inline_paren, s)

    # Step 6: Wrap and heal any bare math that was not enclosed in delimiters
    s = wrap_bare_math(s)

    # Step 6.5: Heal equations, multiplication boundaries, and fraction units
    s = heal_equation_multiplication(s)

    # Step 6.6: Heal student readability (unwrap mistakenly wrapped prose, un-mash words and variables)
    s = heal_student_readability(s)

    # Step 7: Restore HTML tags
    for idx in reversed(range(len(html_tags))):
        s = s.replace(f"\uE000HTMLTAG{idx}\uE001", html_tags[idx])

    return s


def is_indic_text(text: str) -> bool:
    """Returns True if the text contains any Indic characters (Devanagari, Malayalam, Tamil, etc.)."""
    if not text or not isinstance(text, str):
        return False
    return bool(re.search(r'[\u0900-\u0DFF]', text))