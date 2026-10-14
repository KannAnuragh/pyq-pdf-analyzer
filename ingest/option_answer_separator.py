"""
Permanent Option & Answer Separator for PDF Q&A Pipeline.

Designed to reliably separate question stems, multiple-choice options (A, B, C, D),
marked correct answers, and explanations/solutions across 10,000+ PDFs of varying formats
(CBSE, JEE, NEET, State Boards, University Exams, Scanned DPPs, Malayalam & English).

Guarantees:
1. Trailing answer markers ('Answer:', 'Ans:', 'Correct Option:', 'ഉത്തരം:') are NEVER trapped inside options.
2. Trailing solution markers ('Solution:', 'Explanation:', 'വിശദീകരണം:') are separated into the explanation field.
3. Handles attached LaTeX formulas (e.g. 'Answer:\n(b)\\frac{b}{a}' or '(c)$...$') without missing matches.
4. Option prefixes like '(a)', 'A.', 'A)' are stripped from option values.
5. If the answer is printed as a formula/value without a letter (e.g. 'Answer: \\frac{b}{a}'),
   it automatically matches it against option choices to determine the correct letter.
6. Re-classifies questions with valid option sets (A, B, C, D) as 'mcq' rather than 'descriptive'.
"""

import re
from typing import Optional, Dict, Tuple, Any


def _plain_classification_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r'<(?:strong|b|em|i|mark)\b[^>]*>([\s\S]*?)</(?:strong|b|em|i|mark)>', r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r'</?[^>]+>', '', text)
    text = re.sub(r'(?<!\w)(?:\*{1,3}|_{1,3})([^\n]*?)(?:\*{1,3}|_{1,3})(?!\w)', r'\1', text)
    return text.strip()

# Pattern matching Answer markers:
# Must be at the start of a line/string, OR separated by whitespace/tabs with an explicit ':' or '-' delimiter.
# Never match a bare word followed only by newline unless at the start of a line.
# Also allows being preceded by HTML tags like > or token ends like ]]
ANSWER_MARKER_PATTERN = re.compile(
    r"(?:(?:^|\n|\\n|>|\}|\]|\.\s+)\s*(?:\*\*\s*(?:Correct(?:\s*Option|\s*Answer)?|Answer|Ans(?:wer)?|ഉത്തരം|ശരിയായ\s*ഉത്തരം)\s*\*\*|\*?\*?(?:Correct(?:\s*Option|\s*Answer)?|Answer|Ans(?:wer)?|ഉത്തരം|ശരിയായ\s*ഉത്തരം)\*?\*?\s*(?:[\:\-]+\s*|(?=\n|\\n|$))))|"
    r"(?:(?:[ \t]+|\t)(?:\*\*\s*(?:Correct(?:\s*Option|\s*Answer)?|Answer|Ans(?:wer)?|ഉത്തരം|ശരിയായ\s*ഉത്തരം)\s*\*\*|\*?\*?(?:Correct(?:\s*Option|\s*Answer)?|Answer|Ans(?:wer)?|ഉത്തരം|ശരിയായ\s*ഉത്തരം)\*?\*?\s*[\:\-]+\s*))",
    re.IGNORECASE
)

# Pattern matching Solution / Explanation markers:
# A true solution marker is a SECTION HEADER.
# It can only be:
# 1. At line start (or preceded by \n, HTML tag > or token ]]) followed by delimiter (:, -, .-) or on its own line before newline:
#    e.g. "\nExplanation:", "\nSolution\n", "\n**Explanation:**", "\nവിശദീകരണം:"
# 2. OR separated by 2+ spaces or tabs with explicit delimiter (:, -, .-)
# NEVER allow a single space in the middle of a sentence without delimiter, because common exam words like
# "... Reason is correct explanation of Assertion" will falsely match and destroy options and explanations!
SOLUTION_MARKER_PATTERN = re.compile(
    r"(?:(?:^|\n|\\n|>|\}|\]|\.\s+)\s*(?:\*\*\s*(?:Solution|Explanation|Description|Sol|Exp|Hint|Hints|Reasoning|വിശദീകരണം|പരിഹാരം|വിവരണം|സൂചന)\s*\*\*|\*?\*?(?:Solution|Explanation|Description|Sol|Exp|Hint|Hints|Reasoning|വിശദീകരണം|പരിഹാരം|വിവരണം|സൂചന)\*?\*?\s*(?:[\:\.\-]+\s*|(?=\n|\\n|$))))|"
    r"(?:(?:[ \t]{2,}|\t)(?:\*\*\s*(?:Solution|Explanation|Description|Sol|Exp|Hint|Hints|Reasoning|വിശദീകരണം|പരിഹാരം|വിവരണം|സൂചന)\s*\*\*|\*?\*?(?:Solution|Explanation|Description|Sol|Exp|Hint|Hints|Reasoning|വിശദീകരണം|പരിഹാരം|വിവരണം|സൂചന)\*?\*?\s*[\:\.\-]+\s*))",
    re.IGNORECASE
)

# Pattern for option letter at start of answer tail (supports e.g. (b), [b], b, Option B, etc., followed by math/text)
ANSWER_LETTER_PATTERN = re.compile(
    r"^(?:Option\s*[:\-]*\s*)?(?:\(\s*([A-Ha-h1-5])\s*\)|\[\s*([A-Ha-h1-5])\s*\]|([A-Ha-h1-5]))(?=\s|[\.\:\-\$\\\\]|$)",
    re.IGNORECASE
)


def validate_and_repair_latex(text: str) -> Tuple[str, bool]:
    """
    Detects and best-effort repairs structurally broken LaTeX in a field (option text,
    stem, or explanation). This does NOT try to guess the "correct" mathematical meaning —
    it only fixes mechanical brace/delimiter imbalance and reports whether it had to.

    Two checks, both cheap and reliable:
    1. Odd number of unescaped '$' delimiters -> a math span was closed early (or never
       opened), spilling raw LaTeX tokens (\\frac, ^, _, {, }) outside math mode. If the
       text after the last '$' still looks like LaTeX, we fold it back into the span and
       close it properly. Otherwise we just add a closing '$'.
    2. Global brace imbalance ('{' count != '}' count) anywhere in the string. We trim
       excess trailing '}' or append missing '}' — a safe, reversible repair that at least
       stops mismatched braces from rendering as broken LaTeX in production.

    Returns (possibly_repaired_text, had_issue). `had_issue=True` means the field should be
    flagged for human/LLM re-review even after the mechanical repair, because a broken
    delimiter is a symptom of an upstream OCR/LLM extraction error, not just cosmetic.
    """
    if not text or "$" not in text:
        return text, False

    had_issue = False
    fixed = text

    try:
        from .math_normalizer import heal_severed_and_swallowed_math
        fixed = heal_severed_and_swallowed_math(fixed)
    except Exception:
        pass

    dollar_count = fixed.count("$")
    if dollar_count % 2 != 0:
        had_issue = True
        last_dollar = fixed.rfind("$")
        tail = fixed[last_dollar + 1:]
        if re.search(r"\\[a-zA-Z]+|[\^_{}]", tail):
            # Tail still contains LaTeX-looking tokens -> it was meant to stay inside math
            # mode; the '$' closed early. Fold the tail back in and close properly.
            fixed = fixed[:last_dollar] + tail
            fixed = fixed.rstrip() + "$"
        else:
            fixed = fixed + "$"

    open_b = fixed.count("{")
    close_b = fixed.count("}")
    if open_b != close_b:
        had_issue = True
        if close_b > open_b:
            excess = close_b - open_b
            for _ in range(excess):
                idx = fixed.rfind("}")
                if idx == -1:
                    break
                fixed = fixed[:idx] + fixed[idx + 1:]
        else:
            fixed = fixed + ("}" * (open_b - close_b))

    return fixed, had_issue


def _has_subquestion_markers(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    subsequent_b = bool(re.search(r'\S+[\s\S]+?(?:^|\n|\s)(?:\(?[b-hB-H]\)?|[b-hB-H]\.|\(ii\)|\(2\))\s+', text))
    has_a = bool(re.search(r'(?:^|\n|\s)\(?[aA1]\)?[\s\.\:\-]+', text))
    has_b = bool(re.search(r'(?:^|\n|\s)(?:\(?[b-hB-H2]\)?|[b-hB-H]\.)[\s\.\:\-]+', text))
    return subsequent_b or (has_a and has_b)


def parse_trailing_meta(text: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Parses trailing metadata from text (e.g. at the end of an option, question stem, or block).
    Detects marked answer (English 'Answer: B', 'Ans:- B', Malayalam 'ഉത്തരം: B', etc.)
    and solution/explanation ('Solution:', 'Explanation:', 'വിശദീകരണം:', etc.).
    Returns (clean_text, marked_answer, explanation).
    """
    if not text or not isinstance(text, str):
        return "", None, None

    clean_text = text.strip()
    marked_answer = None
    explanation = None

    ans_m = ANSWER_MARKER_PATTERN.search(clean_text)
    sol_m = SOLUTION_MARKER_PATTERN.search(clean_text)

    if ans_m and sol_m:
        if ans_m.start() < sol_m.start():
            # Answer appears first, then Solution
            ans_part = clean_text[ans_m.end():sol_m.start()].strip()
            explanation = clean_text[sol_m.end():].strip()
            clean_text = clean_text[:ans_m.start()].strip()
            has_multi_sub = _has_subquestion_markers(ans_part) or _has_subquestion_markers(explanation) or len(re.findall(r'(?:^|\n|\s)\(([a-d])\)[\s\:\-\.]+', ans_part, re.IGNORECASE)) >= 2
            if not has_multi_sub:
                l_match = ANSWER_LETTER_PATTERN.match(ans_part)
                if l_match:
                    marked_answer = (l_match.group(1) or l_match.group(2) or l_match.group(3)).upper()
                    extra_ans = re.sub(r"^[\s\.\:\-]+", "", ans_part[l_match.end():]).strip()
                    if extra_ans:
                        explanation = f"{extra_ans}\n{explanation}" if explanation else extra_ans
                elif ans_part:
                    explanation = f"{ans_part}\n{explanation}" if explanation else ans_part
            else:
                explanation = f"{ans_part}\n{explanation}" if explanation else ans_part
        else:
            # Solution appears first, then Answer
            explanation = clean_text[sol_m.end():ans_m.start()].strip()
            ans_part = clean_text[ans_m.end():].strip()
            clean_text = clean_text[:sol_m.start()].strip()
            has_multi_sub = _has_subquestion_markers(ans_part) or _has_subquestion_markers(explanation) or len(re.findall(r'(?:^|\n|\s)\(([a-d])\)[\s\:\-\.]+', ans_part, re.IGNORECASE)) >= 2
            if not has_multi_sub:
                l_match = ANSWER_LETTER_PATTERN.match(ans_part)
                if l_match:
                    marked_answer = (l_match.group(1) or l_match.group(2) or l_match.group(3)).upper()
            else:
                explanation = f"{explanation}\n{ans_part}" if explanation else ans_part
    elif ans_m:
        tail = clean_text[ans_m.end():].strip()
        clean_text = clean_text[:ans_m.start()].strip()
        has_multi_sub = _has_subquestion_markers(tail) or len(re.findall(r'(?:^|\n|\s)\(([a-d])\)[\s\:\-\.]+', tail, re.IGNORECASE)) >= 2
        if not has_multi_sub:
            l_match = ANSWER_LETTER_PATTERN.match(tail)
            if l_match:
                marked_answer = (l_match.group(1) or l_match.group(2) or l_match.group(3)).upper()
                exp_part = re.sub(r"^[\s\.\:\-]+", "", tail[l_match.end():]).strip()
                if exp_part:
                    explanation = exp_part
            else:
                explanation = tail
        else:
            explanation = tail
    elif sol_m:
        explanation = clean_text[sol_m.end():].strip()
        clean_text = clean_text[:sol_m.start()].strip()
        ans_in_sol = re.search(
            r"(?:Option\s*[:\-]*\s*\(?([A-E1-5])\)?\s*is\s*correct|\(?([A-E1-5])\)?\s*is\s*(?:the\s*)?correct\s*(?:option|answer))",
            explanation,
            re.IGNORECASE
        )
        if ans_in_sol:
            marked_answer = (ans_in_sol.group(1) or ans_in_sol.group(2)).upper()
        elif explanation and not _has_subquestion_markers(explanation):
            leading_ans = re.match(r"^\(?([A-E1-5])\)?(?=[^a-zA-Z0-9]|$)", explanation, re.IGNORECASE)
            if leading_ans:
                marked_answer = leading_ans.group(1).upper()

    if explanation:
        # Clean redundant labels from start of explanation ONLY when followed by a delimiter or newline.
        # NEVER strip when followed by ordinary prose like "Explanation of Assertion" -> "of Assertion"
        explanation = re.sub(
            r"^(?:Solution|Explanation|Description|Sol|Exp|Hint|വിശദീകരണം|പരിഹാരം|വിവരണം|സൂചന)\s*(?:[\*\:\-]+(?:\s*|\n|$)|(?:\n|\\n))",
            "",
            explanation,
            flags=re.IGNORECASE
        ).strip()
        has_subparts = _has_subquestion_markers(explanation)
        if marked_answer and not has_subparts:
            explanation = re.sub(rf"^\(?{re.escape(marked_answer)}\)?[\s\.\:\-]+", "", explanation, flags=re.IGNORECASE).strip()
            if explanation.strip("() \n\t.").upper() == marked_answer.upper():
                explanation = None
        elif has_subparts:
            if marked_answer == "A":
                marked_answer = None
            starts_with_subpart = bool(re.match(r'^\(?[a-dA-D1-9]\)?[\s\.\:\-]+|^\(?[ivxIVX]+\)?[\s\.\:\-]+', explanation))
            if not starts_with_subpart:
                explanation = f"(a) {explanation}"

    if not marked_answer:
        # Check if text ends with a naked answer like [A] or (C) on its own line
        naked_pat = re.compile(r'(?:^|\n)\s*[\[\(]([A-E1-5])[\]\)]\s*$', re.IGNORECASE)
        m_naked = naked_pat.search(clean_text)
        if m_naked:
            marked_answer = m_naked.group(1).upper()
            clean_text = clean_text[:m_naked.start()].strip()

    return clean_text, marked_answer, explanation


def split_merged_options(options: Dict[str, Any]) -> Dict[str, str]:
    """
    Splits options where two or more options were accidentally merged into one
    (e.g., options['A'] contains '(B) second option' or 'B. second option').
    Guarantees:
    - Splits 'A' into 'A' and 'B', 'B' into 'B' and 'C', etc.
    - Handles multiple merged options in a single string (e.g. (B)... (C)... (D)...).
    - Preserves case format (upper or lower).
    - Avoids false triggers on regular prose words or abbreviations (e.g., 'B.Sc.', 'Beta').
    """
    if not options or not isinstance(options, dict):
        return options or {}

    out: Dict[str, str] = {}
    keys = list(options.keys())

    for k in keys:
        v = options[k]
        if not isinstance(v, str):
            out[k] = str(v) if v is not None else ""
            continue

        current_k = k
        current_text = v.strip()

        # Iteratively look for the expected next option marker in current_text
        while True:
            is_digit = str(current_k).isdigit()
            if is_digit:
                next_val = int(current_k) + 1
                cand_chars = [str(next_val)] if next_val <= 8 else []
                bare_chars = cand_chars
            elif len(str(current_k)) == 1 and str(current_k).isalpha():
                next_ord = ord(str(current_k).upper()) + 1
                cand_char = chr(next_ord)
                if next_ord <= ord('H'):
                    cand_chars = [cand_char, cand_char.lower()]
                    bare_chars = [cand_char] if current_k.isupper() else [cand_char.lower()]
                else:
                    cand_chars = []
                    bare_chars = []
            else:
                cand_chars = []
                bare_chars = []

            if not cand_chars:
                out[current_k] = current_text
                break

            escaped_parens = [re.escape(c) for c in cand_chars]
            paren_group = "|".join(escaped_parens)
            escaped_bares = [re.escape(c) for c in bare_chars]
            bare_group = "|".join(escaped_bares) if bare_chars else ""

            # Look for:
            # 1. Parenthesized/bracketed: (B), [B], (b), [b] (with optional : - .)
            # 2. Bare letter: B., B), B:- (preceded by space/newline, with MANDATORY whitespace after to avoid ratios like c:1)
            bare_branch = rf'|([{bare_group}])[\.\)\:-]\s+' if bare_group else ''
            pat = re.compile(
                rf'(?:(?<=\s)|(?<=\n)|^)(?:\(([{paren_group}])\)[\.\:\-]?\s*|\[([{paren_group}])\][\.\:\-]?\s*{bare_branch})'
            )
            m = pat.search(current_text)
            if m and m.start() > 0:
                split_idx = m.start()
                first_part = current_text[:split_idx].strip()
                if not first_part:
                    out[current_k] = current_text
                    break
                matched_char = m.group(1) or m.group(2) or (m.group(3) if len(m.groups()) >= 3 else None)
                next_k = matched_char.upper() if current_k.isupper() else matched_char.lower()
                second_part = current_text[m.end():].strip()

                out[current_k] = first_part
                current_k = next_k
                current_text = second_part
            else:
                out[current_k] = current_text
                break

    return out


def separate_options_dict(
    options: Dict[str, Any],
    marked_answer: Optional[str] = None,
    explanation: Optional[str] = None
) -> Tuple[Dict[str, str], Optional[str], Optional[str]]:
    """
    Cleans an options dictionary, extracting any trailing answer keys or solutions
    accidentally embedded in options (e.g. Option D containing 'Answer:\n(b) b/a').
    """
    clean_opts: Dict[str, str] = {}
    extracted_ans = marked_answer
    extracted_exp = explanation

    options = split_merged_options(options)

    for k, v in options.items():
        if not isinstance(v, str):
            clean_opts[k] = str(v) if v is not None else ""
            continue

        # Strip accidental leading option prefixes like "(A) ", "A. ", "a: "
        # CRITICAL: Must be in parens/brackets or followed by delimiter/space.
        # NEVER strip 'B' from 'Both' or 'A' from 'All'!
        k_esc = re.escape(k)
        # Only strip option prefix when it's clearly a label marker (in parens/brackets or
        # followed by delimiter), NEVER when the next character forms a word continuation.
        # e.g. "(b) Both" strips "(b) " but NOT the "B" from "Both".
        # e.g. "b. Both" strips "b. " but NOT the "B" from "Both".
        # The key insight: after stripping the label, if the remaining text starts with a
        # lowercase letter that could be the continuation of a word, don't strip.
        pat = rf"^\s*(?:\([\s]*[{k_esc}{k_esc.lower()}][\s]*\)[\.\:\-]?|\[[\s]*[{k_esc}{k_esc.lower()}][\s]*\][\.\:\-]?|[{k_esc}{k_esc.lower()}]\)[\.\:\-]?)\s+"
        v_clean = re.sub(pat, "", v.strip())
        # Auto-heal truncated words caused by historical/upstream faulty stripping
        v_clean = re.sub(r'\bOth\b', 'Both', v_clean)
        v_clean = re.sub(r'^(?:[oO]th)\b', 'Both', v_clean)
        def _fix_both_opt(m: re.Match) -> str:
            idx = m.start()
            if idx == 0 or re.search(r'(?:^|[.!?\n]\s*)$', v_clean[:idx]):
                return 'Both'
            return 'both'
        v_clean = re.sub(r'\both\b', _fix_both_opt, v_clean)
        v_clean = re.sub(r'\bll\s+of\s+the\s+above\b', 'All of the above', v_clean, flags=re.IGNORECASE)

        # Strip marks indicators from option text (e.g. '[1 mark]', '(2 marks)', '[5 marks]')
        v_clean = re.sub(r'(?i)\s*\[\s*\d+(?:\.\d+)?(?:\s*/\s*\d+)?\s*(?:marks?|mark|m)\s*\]', '', v_clean)
        v_clean = re.sub(r'(?i)\s*\(\s*\d+(?:\.\d+)?(?:\s*/\s*\d+)?\s*(?:marks?|mark|m)\s*\)', '', v_clean)
        v_clean = v_clean.strip()

        # Separate trailing answer or explanation from option body
        opt_body, ans, exp = parse_trailing_meta(v_clean)
        clean_opts[k] = opt_body

        if ans and not extracted_ans:
            extracted_ans = ans
        if exp and not extracted_exp:
            extracted_exp = exp
    # Extract answer if the explanation explicitly starts with an option letter (e.g., "(C) E != 0" or "Answer: A")
    if extracted_exp:
        has_multi_sub_exp = len(re.findall(r'(?:^|\n)\s*(?:\([a-zA-Z0-9]+\)|[a-zA-Z0-9]+[\.\)\:-])\s+', extracted_exp)) >= 2
        if not has_multi_sub_exp:
            pat1 = r"^\s*(?:Answer|Ans|Solution|Sol|Correct\s*Option|Option|Key)[\s\:\-\.]*[\(\[]?\s*([A-E1-5])\s*[\)\]]?(?:[\s\.\:\-]+|is\s*correct\b)?"
            pat2 = r"^\s*(?:[\(\[]\s*([A-E1-5])\s*[\)\]]|([A-E1-5])\s*[\.\:\-\)]+)(?=\s|$)"
            pat3 = r"^\s*([A-E1-5])\s*$"
            
            m1 = re.match(pat1, extracted_exp, re.IGNORECASE)
            m2 = re.match(pat2, extracted_exp, re.IGNORECASE)
            m3 = re.match(pat3, extracted_exp, re.IGNORECASE)
            
            explicit_match = m1 or m2 or m3
            if explicit_match:
                candidate = (explicit_match.group(1) if m1 else (m2.group(1) or m2.group(2) if m2 else m3.group(1))).upper()
                # Only override if this candidate actually exists in the options
                if any(k.upper() == candidate for k in clean_opts.keys()):
                    # Only override if it's an explicit "Answer:" marker OR if we don't already have an answer
                    if m1 or not extracted_ans:
                        extracted_ans = candidate
        else:
            # If multi-step explanation, only match explicit Answer: prefix
            pat1 = r"^\s*(?:Answer|Ans|Solution|Sol|Correct\s*Option|Option|Key)[\s\:\-\.]*[\(\[]?\s*([A-E1-5])\s*[\)\]]?(?:[\s\.\:\-]+|is\s*correct\b)?"
            m1 = re.match(pat1, extracted_exp, re.IGNORECASE)
            if m1:
                candidate = m1.group(1).upper()
                if any(k.upper() == candidate for k in clean_opts.keys()):
                    extracted_ans = candidate

    # If marked answer wasn't explicitly given as a letter, check if explanation/answer text
    # or its first line matches one of the option values (e.g. Answer: b/a matches Option B)
    if not extracted_ans and extracted_exp:
        first_line = extracted_exp.splitlines()[0] if extracted_exp else ""
        first_clean = first_line.strip("$\n\t ").replace(" ", "")
        exp_clean = extracted_exp.strip("$\n\t ").replace(" ", "")
        matched_k = None
        matched_by_first_line = False

        for k, v in clean_opts.items():
            v_clean = v.strip("$\n\t ").replace(" ", "")
            if v_clean:
                if exp_clean == v_clean:
                    matched_k = k.upper()
                    break
                elif first_clean and first_clean == v_clean:
                    matched_k = k.upper()
                    matched_by_first_line = True
                    break

        if matched_k:
            extracted_ans = matched_k
            if matched_by_first_line:
                remaining_lines = extracted_exp.splitlines()[1:]
                extracted_exp = "\n".join(remaining_lines).strip() or None

    clean_opts = split_merged_options(clean_opts)
    return clean_opts, extracted_ans, extracted_exp


def is_valid_subparts(sub_parts: Optional[Dict[str, Any]], stem: Optional[str] = None) -> bool:
    """
    Validates whether an extracted sub_parts dict represents genuine subquestions.
    Rules:
    1. Must have at least 2 parts.
    2. Sequence must start from 'a', '1', or 'i' (never start from 'b' or isolated letters).
    3. Keys must follow strict ascending sequence: ('a', 'b'), ('1', '2'), ('i', 'ii').
    4. No part can be empty or whitespace-only.
    5. No part can contain ONLY marks indicators (e.g. {"b": "[2 marks]"} is invalid).
    6. Each part must have substantive question text (at least 3 word characters).
    7. ZERO ANSWER LABELING AS SUBQUESTION: Solution steps, values, comparative states
       (Higher, Lower, Increases, Decreases, R_T = 20 Ohm) are ANSWERS, NOT SUBQUESTIONS!
       At least one part MUST contain a genuine question directive (define, calculate, state,
       derive, draw, explain, what, why, how, etc. in English or Malayalam), a question mark (?),
       explicit marks allocation, OR the introductory question stem provides the directive prompt
       (e.g. "Explain how ... when:"). Pure answer steps are strictly rejected.
    """
    if not sub_parts or not isinstance(sub_parts, dict):
        return False
    if len(sub_parts) < 2:
        return False

    keys = sorted([str(k).lower() for k in sub_parts.keys()])
    first_key = keys[0]
    if first_key not in ('a', '1', 'i'):
        return False

    # Check consecutive sequence
    if first_key == 'a':
        expected = [chr(ord('a') + i) for i in range(len(keys))]
        if keys != expected:
            return False
    elif first_key == '1':
        expected = [str(i + 1) for i in range(len(keys))]
        if keys != expected:
            return False
    elif first_key == 'i':
        roman_seq = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii']
        if keys != roman_seq[:len(keys)]:
            return False

    marks_only_pat = re.compile(
        r'^\s*(?:\[\s*\d+(?:\.\d+)?(?:\s*/\s*\d+)?\s*(?:marks?|mark|m)\s*\]|\(\s*\d+(?:\.\d+)?(?:\s*/\s*\d+)?\s*(?:marks?|mark|m)\s*\)|\b\d+\s*(?:marks?|mark|m)\b|[\d\.\:\-\s])*\s*$',
        re.IGNORECASE
    )

    eng_directives = (
        'define', 'explain', 'calculate', 'draw', 'state', 'derive', 'determine',
        'find', 'show that', 'show', 'prove that', 'prove', 'distinguish',
        'differentiate', 'obtain', 'evaluate', 'name', 'give', 'write',
        'what', 'why', 'how', 'which', 'where', 'when', 'who', 'compare',
        'list', 'mention', 'describe', 'illustrate', 'sketch', 'identify',
        'plot', 'compute', 'verify'
    )
    mal_directives = (
        'എഴുതുക', 'വിശദീകരിക്കുക', 'കണക്കാക്കുക', 'വരയ്ക്കുക', 'നിർവചിക്കുക',
        'കണ്ടെത്തുക', 'തെളിയിക്കുക', 'വ്യത്യാസ', 'രൂപീകരിക്കുക', 'പ്രസ്താവിക്കുക',
        'താരതമ്യം', 'പട്ടികപ്പെടുത്തുക', 'സൂചിപ്പിക്കുക', 'എന്താണ്', 'എന്ത്',
        'എങ്ങനെ', 'എവിടെ', 'എപ്പോൾ', 'ഏതാണ്', 'വിവരിക്കുക', 'തരംതിരിക്കുക'
    )

    stem_has_directive = False
    if stem and isinstance(stem, str):
        stem_strip = _plain_classification_text(stem)
        stem_low = stem_strip.lower()
        stem_has_directive = bool(
            re.search(r'^(?:explain|state|describe|calculate|find|define|show|prove|derive|write|determine|what|why|how|distinguish|differentiate|obtain|evaluate|name|give|list|mention|compare|വിശദീകരിക്കുക|കണക്കാക്കുക|എഴുതുക|നിർവചിക്കുക|കണ്ടെത്തുക|തെളിയിക്കുക)\b', stem_low)
            or stem_strip.endswith((':', 'when:', 'following:', 'if:', 'below:', 'എങ്കിൽ:'))
        )

    has_question_feature = stem_has_directive
    valid_count = 0

    for k, v in sub_parts.items():
        v_str = _plain_classification_text(v)
        if not v_str:
            return False
        # Remove html tags like <img .../> and <div ...> to inspect actual question text
        text_only = re.sub(r'<[^>]+>', '', v_str).strip()
        if not text_only or marks_only_pat.match(text_only):
            return False

        t_low = text_only.lower()

        # Reject pure answer derivations / comparative states without question directives
        is_pure_answer_val = bool(re.match(r'^(?:higher|lower|increases?|decreases?|remains constant|constant|zero|positive|negative|forward bias|reverse bias|same|remains same|no change)[\.\s]*$', t_low))
        has_answer_header = any(ans in t_low for ans in ('answer:', 'ans:', 'solution:', 'sol:', 'hence', 'therefore', 'we get'))
        has_multi_eq = '=' in text_only and len(text_only.split('=')) >= 2

        # Check for directive question features:
        has_qmark = '?' in text_only
        has_marks = bool(re.search(r'\[\s*\d+\s*marks?\s*\]|\(\s*\d+\s*marks?\s*\)|\b\d+\s*marks?\b', text_only, re.IGNORECASE))
        has_directive = any(re.match(r'^\s*' + re.escape(d) + r'\b', t_low) for d in eng_directives) or any(t_low.startswith(m) for m in mal_directives) or (has_marks and any(re.search(r'\b' + re.escape(d) + r'\b', t_low) for d in eng_directives))

        if has_qmark or has_marks or has_directive:
            has_question_feature = True
        elif is_pure_answer_val or has_answer_header or has_multi_eq:
            # Answer step with no question directive is NEVER a subquestion!
            return False

        # Must have some alphabetical / Indic characters
        if len(re.findall(r'[a-zA-Z\u0D00-\u0D7F]', text_only)) >= 3:
            valid_count += 1

    return (valid_count >= 2) and has_question_feature


def is_equation_or_formula_text(text: Any) -> bool:
    """Detects if an option text is predominantly a mathematical formula, equation, value, or unit."""
    if not text:
        return True
    t = str(text).strip()
    if not t:
        return True
    if "$" in t or r"\frac" in t or r"\sqrt" in t or r"\times" in t or r"\mu" in t or r"\pi" in t or r"\vec" in t or r"\hat" in t:
        return True
    if re.search(r'[\=\+\-\*\/\^\_\±\×\÷\√\∮\∴\⇒\σ\ε\ϕ\π\μ\τ°²³⁻¹⁰]', t):
        return True
    stripped = re.sub(r'<[^>]+>', '', t)
    stripped = re.sub(r'[\d\s\=\+\-\*\/\^\_{\}\(\)\[\]\\\.,\:\;\%×÷±√πσεϕθμτ°²³⁻¹⁰]', '', stripped)
    if len(stripped) < 4:
        return True
    return False


def strip_and_extract_trailing_options_from_stem(
    stem: str,
    options: Optional[Dict[str, Any]] = None,
    marked_answer: Optional[str] = None,
    explanation: Optional[str] = None
) -> Tuple[str, Dict[str, str], Optional[str], Optional[str]]:
    """
    Universally extracts trailing options and answers trapped at the bottom of the stem.
    Guarantees:
    1. Strips trailing options (e.g. '\n(D) 1.0 × 10' or ' (D) 1.0 × 10') and adds to options['D'].
    2. Iteratively extracts preceding options (e.g. (C)... (D)...) in reverse order.
    3. Strips trailing marks indicators like '[1 mark]', '(2 marks)'.
    4. If explanation contains 'Answer: (b) 5.0 \times 10^{17}', extracts marked_answer='B'
       and populates options['B'] = '5.0 \times 10^{17}' if missing.
    """
    opts: Dict[str, str] = {str(k): str(v) for k, v in (options or {}).items()}
    s = (stem or "").strip()
    cur_ans = marked_answer
    cur_exp = explanation

    # 1. Strip trailing marks bracket from stem first
    s = re.sub(
        r'(?i)\s*(?:\[\s*\d+(?:\.\d+)?(?:\s*/\s*\d+)?\s*(?:marks?|mark|m)\s*\]|\(\s*\d+(?:\.\d+)?(?:\s*/\s*\d+)?\s*(?:marks?|mark|m)\s*\))\s*$',
        '',
        s
    ).strip()

    # 2. Trailing single/multiple option stripping from stem (ONLY if options are non-empty and incomplete)
    # If opts is empty, a descriptive question with math equations must NEVER have pieces sliced off its stem!
    if 0 < len(opts) < 4:
        trailing_opt_pat = re.compile(
            r'(?:(?:\n|\s+|\.\s+)\(([A-Ea-e1-4])\)[\.\:\-]?\s*|(?:\n|\s{2,}|\.\s+)([A-Ea-e1-4])[\.\)\:-]\s*(?!\d))([^ \n\r\t][^\n\r]*)$',
            re.IGNORECASE
        )

        while True:
            m = trailing_opt_pat.search(s)
            if not m:
                break
            letter = (m.group(1) or m.group(2)).upper()
            val = m.group(3).strip()
            start_idx = m.start()
            prefix = s[max(0, start_idx - 35):start_idx].lower()

            has_paren = bool(m.group(1))

            # Reject if followed by equation operator or equal sign (e.g. A = 240, A = 120, A + B)
            # Parenthesized options like (d) \sqrt{c}: 1 or (b) +5 V are completely valid!
            if not has_paren and re.match(r'^[\=\+\-\*\/\^\_\±\×\÷\√]', val):
                break
            elif has_paren and val.startswith('='):
                break

            # Reject candidate matches preceded by Assertion/Reason/Statement labels:
            if re.search(r'(?:assertion|reason|statement|പ്രസ്താവന|കാരണം|point|points|junction|terminal|cell|cells|charge|charges|plate|plates|number|mass|each|fragment|fragments|side|radius|amplitude)[\s\:\-\*\_\(]*$', prefix, re.IGNORECASE):
                break
            # Reject if val contains Reason/Assertion label
            if re.search(r'\b(?:reason|കാരണം|assertion|പ്രസ്താവന)\b', val, re.IGNORECASE):
                break
            # If letter already in opts with non-empty content, never overwrite or slice
            if letter in opts and opts[letter].strip():
                break

            if s[:start_idx].strip():
                s = s[:start_idx].strip()
                s = re.sub(
                    r'(?i)\s*(?:\[\s*\d+(?:\.\d+)?(?:\s*/\s*\d+)?\s*(?:marks?|mark|m)\s*\]|\(\s*\d+(?:\.\d+)?(?:\s*/\s*\d+)?\s*(?:marks?|mark|m)\s*\))\s*$',
                    '',
                    s
                ).strip()
                if re.match(r"^(?:question\s*\d*|q\d*|ചോദ്യം\s*\d*|[\d\.\:\-\s])*$", s, re.IGNORECASE):
                    s = ""
                opts[letter] = val
            else:
                s = ""
                opts[letter] = val

    # 3. If explanation has explicit answer like "Answer: (b) 5.0 \times 10^{17}"
    if cur_exp and isinstance(cur_exp, str):
        ans_match = re.search(
            r'(?:^|\n|\s{2,}|\b)(?:\*\*\s*(?:Answer|Ans(?:wer)?|Correct(?:\s*Option|\s*Answer)?|ഉത്തരം)\s*\*\*|\*?\*?(?:Answer|Ans(?:wer)?|Correct(?:\s*Option|\s*Answer)?|ഉത്തരം)\*?\*?\s*[\:\.\-]+)\s*[\(\[]?\s*([A-Ea-e1-5])\s*[\)\]]?[\s\:\.\-]*([^\n]*)',
            cur_exp,
            re.IGNORECASE
        )
        if ans_match:
            ans_letter = ans_match.group(1).upper()
            ans_text = ans_match.group(2).strip()
            if not cur_ans:
                cur_ans = ans_letter
            if ans_text and (ans_letter not in opts or not opts[ans_letter]):
                opts[ans_letter] = ans_text
        elif not cur_ans:
            leading_m = re.match(r'^\s*[\(\[]\s*([A-Ea-e1-5])\s*[\)\]][\s\:\.\-]*([^\n]*)', cur_exp, re.IGNORECASE)
            if leading_m:
                ans_letter = leading_m.group(1).upper()
                ans_text = leading_m.group(2).strip()
                cur_ans = ans_letter
                if ans_text and (ans_letter not in opts or not opts[ans_letter]):
                    opts[ans_letter] = ans_text

    return s, opts, cur_ans, cur_exp


def classify_question_type(
    options: Dict[str, Any],
    explanation: Optional[str] = None,
    marked_answer: Optional[str] = None,
    stem: Optional[str] = None
) -> Tuple[str, Optional[Dict[str, str]]]:
    """
    Determines if a question is MCQ or descriptive (single or with subquestions).
    
    MCQ: options are standard choices (A, B, C, D) with answer choices (formulas, values, statements, equations).
    Descriptive:
      - Has sub-questions: options are genuine subproblems like (a), (b) asking students to explain, calculate, derive, define
      - Or contains question marks / marks indicators [1 mark]
      - Or options are empty / single open-ended question.
    """
    if not options or len(options) < 2:
        return "descriptive_single", None

    keys_upper = sorted([re.sub(r'[^A-Z0-9]', '', str(k).strip().upper()) for k in options.keys()])
    sub_parts = {str(k).strip().lower(): v for k, v in options.items()}

    # Helper to return descriptive_subquestions ONLY if subparts are genuinely valid
    def _subq_or_single(parts: Dict[str, str]) -> Tuple[str, Optional[Dict[str, str]]]:
        filtered_parts = {
            k: v for k, v in parts.items()
            if str(v).strip().lower() not in (
                "none of the above", "all of the above", "none of these", "all of these", "none", "all"
            )
        }
        target_parts = filtered_parts if len(filtered_parts) >= 2 else parts
        if is_valid_subparts(target_parts, stem=stem):
            return "descriptive_subquestions", target_parts
        return "descriptive_single", None

    # Check if options contain synthetic padding (e.g. "None of the above") on top of multi-variable answer steps
    non_dummy = {
        k: v for k, v in options.items()
        if str(v).strip().lower() not in (
            "none of the above", "all of the above", "none of these", "all of these", "none", "all"
        )
    }
    if len(non_dummy) < len(options) and len(non_dummy) >= 2:
        has_states = any(str(v).strip().lower() in ("higher", "lower", "increase", "decrease", "forward bias", "reverse bias") for v in non_dummy.values())
        has_multi_eq = any("=" in str(v) and len(str(v).split("=")) >= 3 for v in non_dummy.values())
        if has_states and has_multi_eq:
            return "descriptive_single", None

    # Check if options duplicate answer steps in explanation (e.g. 1. Higher, 2. Lower)
    if explanation and isinstance(explanation, str) and (not marked_answer or str(marked_answer).strip().upper() not in keys_upper):
        exp_lines = [line.strip().lower() for line in explanation.splitlines() if line.strip()]
        opt_vals = [str(v).strip().lower() for v in options.values()]
        if any(any(re.match(rf'^(?:[1-9]\.|\([1-9]\)|\([a-d]\))\s*{re.escape(ov)}$', el) for el in exp_lines) for ov in opt_vals if ov in ("higher", "lower", "increases", "decreases", "forward bias", "reverse bias")):
            return "descriptive_single", None

    # If options contain images/diagrams, it is an MCQ!
    if any(isinstance(v, str) and ('<img' in v or '<svg' in v or '{{IMG' in v or '[[QAI_IMAGE' in v) for v in options.values()):
        return "mcq", None

    # ASSERTION-REASON DETECTION: If option text mentions "Assertion" and "Reason" or "Both A and R", force MCQ.
    assertion_reason_signals = 0
    for v in options.values():
        if isinstance(v, str):
            v_low = _plain_classification_text(v).lower()
            if re.search(r'(?:both\s+)?assertion\s+and\s+reason', v_low, re.IGNORECASE):
                assertion_reason_signals += 1
            elif re.search(r'assertion\s+is\s+(?:true|false|correct)', v_low, re.IGNORECASE):
                assertion_reason_signals += 1
            elif re.search(r'\bboth\s+[aA]\s+and\s+[rR]\b', v_low, re.IGNORECASE):
                assertion_reason_signals += 1
            elif re.search(r'\b[aA]\s+is\s+(?:true|false)\b', v_low, re.IGNORECASE):
                assertion_reason_signals += 1
            elif re.search(r'\b[rR]\s+is\s+(?:the\s+)?(?:correct\s+)?explanation\b', v_low, re.IGNORECASE):
                assertion_reason_signals += 1
    if assertion_reason_signals >= 2:
        return "mcq", None

    if stem and isinstance(stem, str):
        stem_low = _plain_classification_text(stem).lower()
        if re.search(r'\b(?:assertion|പ്രസ്താവന)\b', stem_low) and re.search(r'\b(?:reason|കാരണം)\b', stem_low):
            if len(options) >= 2:
                return "mcq", None

    # A/B (or 1/2) directive answers are descriptive subquestions, even when
    # an answer extractor supplied a matching letter. Do this before the
    # generic MCQ-key check below; otherwise two-part questions become MCQs.
    if len(options) in (2, 3) and is_valid_subparts(sub_parts, stem=stem):
        directive_values = [_plain_classification_text(v).lower() for v in options.values()]
        has_directive = any(
            re.match(r'^(?:define|explain|calculate|draw|state|derive|determine|find|show|prove|write|what|why|how|which|describe|compare|evaluate|obtain|വിശദീകരിക്കുക|കണക്കാക്കുക|എഴുതുക|നിർവചിക്കുക|കണ്ടെത്തുക|തെളിയിക്കുക)\b', value)
            for value in directive_values
        )
        has_marks = any(re.search(r'\b\d+\s*marks?\b|\[\s*\d+\s*marks?', value, re.IGNORECASE) for value in directive_values)
        if has_directive or has_marks:
            return "descriptive_subquestions", sub_parts

    # If marked answer is provided and matches an option key, it is an MCQ!
    if marked_answer and str(marked_answer).strip().upper() in keys_upper:
        return "mcq", None

    # Check if question stem is an introductory directive command (e.g. "Explain how ... when:" or "Calculate the following:")
    # introducing sub-tasks (a) and (b) without an answer key
    if stem and isinstance(stem, str) and not marked_answer and len(options) in (2, 3):
        stem_strip = _plain_classification_text(stem)
        has_directive_stem = bool(
            re.search(r'^(?:explain|state|describe|calculate|find|define|show|prove|derive|write|determine|what|why|how|distinguish|differentiate|obtain|evaluate|name|give|list|mention|compare|വിശദീകരിക്കുക|കണക്കാക്കുക|എഴുതുക|നിർവചിക്കുക|കണ്ടെത്തുക|തെളിയിക്കുക)\b', stem_strip, re.IGNORECASE)
            or stem_strip.endswith((':', 'when:', 'following:', 'if:', 'below:', 'എങ്കിൽ:', ':'))
        )
        if has_directive_stem and is_valid_subparts(sub_parts, stem=stem):
            return "descriptive_subquestions", sub_parts

    # ---- Directive sub-question check (MUST run BEFORE generic MCQ key check) ----
    # If option texts look like genuine directive sub-questions (start with imperative
    # commands, contain marks indicators, or have question marks), classify as descriptive
    # BEFORE we even check if keys are A/B/C/D. This prevents (a) Name device, (b) Draw
    # circuit, (c) Explain output, (d) Define from being classified as MCQ.
    mal_directives = (
        'എഴുതുക', 'വിശദീകരിക്കുക', 'കണക്കാക്കുക', 'വരയ്ക്കുക', 'നിർവചിക്കുക',
        'കണ്ടെത്തുക', 'തെളിയിക്കുക', 'വ്യത്യാസ', 'രൂപീകരിക്കുക', 'പ്രസ്താവിക്കുക',
        'താരതമ്യം', 'പട്ടികപ്പെടുത്തുക', 'സൂചിപ്പിക്കുക'
    )
    eng_directives = (
        'draw', 'calculate', 'find', 'explain', 'state', 'prove',
        'derive', 'show that', 'determine', 'define', 'describe',
        'distinguish', 'differentiate', 'obtain', 'evaluate', 'name',
        'what', 'why', 'how', 'which'
    )

    sub_q_count = 0
    has_marks = False
    has_qmark_in_opts = False
    marks_pat = re.compile(r'\[\s*\d+\s*marks?\s*\]|\(\s*\d+\s*marks?\s*\)|\b\d+\s*marks?\b', re.IGNORECASE)

    for v in options.values():
        if isinstance(v, str):
            v_low = _plain_classification_text(v).lower()
            if marks_pat.search(v):
                has_marks = True
            if '?' in v_low:
                has_qmark_in_opts = True
            if any(v_low.startswith(m) for m in mal_directives):
                sub_q_count += 1
            elif any(re.match(r'^\s*' + re.escape(e) + r'\b', v_low) for e in eng_directives):
                sub_q_count += 1

    # If 2+ options are directive sub-questions, OR options have marks indicators,
    # OR options contain question marks — classify as descriptive subquestions.
    if (has_marks or sub_q_count >= 2 or (has_qmark_in_opts and sub_q_count >= 1)) and is_valid_subparts(sub_parts, stem=stem):
        return _subq_or_single(sub_parts)

    # Check if option texts are unreasonably long for MCQ choices (> 350 chars avg of actual text)
    clean_lens = [len(_plain_classification_text(v)) for v in options.values()]
    avg_len = sum(clean_lens) / max(1, len(options))
    if avg_len > 350 and is_valid_subparts(sub_parts, stem=stem):
        return _subq_or_single(sub_parts)

    valid_mcq_sets = (
        ["A", "B", "C", "D"],
        ["A", "B", "C"],
        ["A", "B", "C", "D", "E"],
        ["A", "B"],
        ["1", "2", "3", "4"],
        ["1", "2", "3"],
        ["1", "2"],
        ["I", "II", "III", "IV"],
        ["I", "II", "III"]
    )

    is_valid_digit_mcq = all(k in ('1', '2', '3', '4', '5', '6') for k in keys_upper) and '1' in keys_upper and len(keys_upper) >= 2
    if is_valid_digit_mcq:
        sorted_digits = sorted([int(k) for k in keys_upper])
        if sorted_digits != list(range(1, len(sorted_digits) + 1)):
            is_valid_digit_mcq = False

    is_valid_letter_mcq = all(k in ('A', 'B', 'C', 'D', 'E', 'F') for k in keys_upper) and 'A' in keys_upper and len(keys_upper) >= 2
    if is_valid_letter_mcq:
        sorted_letters = sorted(keys_upper)
        expected = [chr(ord('A') + i) for i in range(len(sorted_letters))]
        if sorted_letters != expected:
            is_valid_letter_mcq = False
    is_valid_roman_mcq = all(k in ('I', 'II', 'III', 'IV', 'V') for k in keys_upper) and 'I' in keys_upper and len(keys_upper) >= 2

    is_valid_mcq = (keys_upper in valid_mcq_sets) or is_valid_digit_mcq or is_valid_letter_mcq or is_valid_roman_mcq

    if not is_valid_mcq:
        return _subq_or_single(sub_parts)

    # If all or most options are formulas/equations/values/units/ratios, force MCQ classification
    # This MUST happen before _subq_or_single to prevent equation MCQ options from being
    # misclassified as subquestions (Bug 4: A,B,C,D with all-equation options -> 'a,b,c,d' subparts)
    formula_count = sum(1 for v in options.values() if is_equation_or_formula_text(v))
    if formula_count >= len(options) - 1 and len(options) >= 2:
        return "mcq", None

    return "mcq", None


def strip_subquestion_labels_from_answer(text: Optional[str]) -> Optional[str]:
    """
    Ensures answers and solution steps are never labeled as subquestions.
    Strips labels like 'Subquestion 1:', 'Sub-question (a):', 'Subquestion:', '(a) Subquestion:' etc.
    while preserving the actual answer content and numbering.
    """
    if not text or not isinstance(text, str):
        return text
    # Replace e.g. "Subquestion 1: Higher" -> "1. Higher", "Subquestion (a): Higher" -> "(a) Higher"
    s = re.sub(
        r'(?i)(?:^|\n|\s+)\*?\*?Sub-?\s*questions?\s*(?:\(([a-zA-Z0-9]+)\)|([a-zA-Z0-9]+))?[\s\:\-\.]*\*?\*?',
        lambda m: (f"\n({m.group(1)}) " if m.group(1) else (f"\n{m.group(2)}. " if m.group(2) else "\n")),
        text
    )
    # Replace e.g. "Sub-part 1: Higher" -> "1. Higher"
    s = re.sub(
        r'(?i)(?:^|\n|\s+)\*?\*?Sub-?\s*parts?\s*(?:\(([a-zA-Z0-9]+)\)|([a-zA-Z0-9]+))?[\s\:\-\.]*\*?\*?',
        lambda m: (f"\n({m.group(1)}) " if m.group(1) else (f"\n{m.group(2)}. " if m.group(2) else "\n")),
        s
    )
    # Clean up excessive newlines
    s = re.sub(r'\n{3,}', '\n\n', s).strip()
    return s


def extract_subparts_from_text(text: str) -> Tuple[Optional[Dict[str, str]], str]:
    """
    Extracts subquestions (e.g. (a), (b), (c) or (i), (ii)) from text
    when they are not already in sub_parts or options.
    Returns a dict mapping part key ('a', 'b', etc.) to the subquestion text,
    or None if no valid subquestion sequence is found.
    Answers and solutions are NEVER scanned or extracted as subquestions.
    """
    if not text or not isinstance(text, str):
        return None, text

    # Strip any trailing answer or solution section before searching for subquestions!
    # Answers must NEVER be scanned or extracted as subquestions!
    clean_stem, _, _ = parse_trailing_meta(text)
    if clean_stem and clean_stem != text:
        text = clean_stem

    ans_m = ANSWER_MARKER_PATTERN.search(text)
    if ans_m:
        text = text[:ans_m.start()].strip()
    sol_m = SOLUTION_MARKER_PATTERN.search(text)
    if sol_m:
        text = text[:sol_m.start()].strip()

    # Matches markers like (a), (b) or (1), (2) or (i), (ii), or bare letters a., b., a), b)
    subq_pattern = re.compile(
        r'(?:(?:^|\n|\s{2,}|\.\s+)\(([a-eA-E1-4]|[ivxlcdmIVXLCDM]+)\)\s*)|'
        r'(?:(?:^|\n|[ \t]{2,}|\.\s+)([a-eA-E]|[ivxlcdmIVXLCDM]+)[\.\)\:-]\s*(?!\d))'
    )
    raw_matches = list(subq_pattern.finditer(text))
    if len(raw_matches) < 2:
        return None, text

    valid_matches = []
    for m in raw_matches:
        k = (m.group(1) or m.group(2)).lower()
        start_idx = m.start(1) if m.group(1) else m.start(2)
        prefix = text[max(0, start_idx - 35):start_idx].lower()
        has_paren = bool(m.group(1))

        # Reject if followed by equation operator or equal sign (e.g. A = 240, A = 120, A + B)
        # Parenthesized markers like (b) +5 V or (c) \sqrt{2} should never be rejected
        post_snippet = text[m.end():m.end() + 10].strip()
        if not has_paren and re.match(r'^[\=\+\-\*\/\^\_\±\×\÷\√]', post_snippet):
            continue
        elif has_paren and post_snippet.startswith('='):
            continue

        # Reject candidate matches preceded by Assertion/Reason/Statement labels:
        if re.search(r'(?:assertion|reason|statement|പ്രസ്താവന|കാരണം)[\s\:\-\*\_\(]*$', prefix, re.IGNORECASE):
            continue
        # Reject candidate matches preceded by prepositions / linkers / points / math variable contexts:
        # ONLY reject if the preceding word is a full word (2+ chars)
        prev_word_m = re.search(r'([a-zA-Z]+)\s*(?:and\s*)?$', prefix)
        if prev_word_m:
            pw = prev_word_m.group(1).lower()
            reject_set = {'both', 'between', 'from', 'to', 'and', 'or', 'is', 'are', 'was', 'were',
                         'if', 'whether', 'that', 'when', 'where', 'of', 'in', 'for', 'with',
                         'junction', 'terminal', 'point', 'points', 'cell', 'cells',
                         'charge', 'charges', 'plate', 'plates', 'number', 'mass',
                         'each', 'fragment', 'fragments', 'side', 'radius', 'amplitude',
                         'wavelength', 'frequency', 'constant', 'variable'}
            if len(pw) >= 2 and pw in reject_set:
                continue
        # Reject candidate matches preceded by digits (e.g. scientific unit "0.2 A.")
        if not has_paren and re.search(r'\d+(?:\.\d+)?\s*$', prefix):
            continue
        valid_matches.append((k, m))

    if len(valid_matches) < 2:
        return None, text

    keys = [item[0] for item in valid_matches]
    first_key = keys[0]
    if first_key not in ('a', '1', 'i'):
        return None, text

    subparts: Dict[str, str] = {}
    for i, (k, m) in enumerate(valid_matches):
        start = m.end()
        end = valid_matches[i + 1][1].start() if i + 1 < len(valid_matches) else len(text)
        part_content = text[start:end].strip()
        # Strip trailing answer or explanation from the last sub-part if present
        if i == len(valid_matches) - 1:
            clean_part, _, _ = parse_trailing_meta(part_content)
            part_content = clean_part
        if part_content:
            subparts[k] = part_content

    if not is_valid_subparts(subparts):
        return None, text

    first_match_start = valid_matches[0][1].start()
    clean_stem = text[:first_match_start].strip()

    return subparts, clean_stem


def clean_stem_contamination(
    stem: str,
    marked_answer: Optional[str] = None,
    explanation: Optional[str] = None
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Strips an "Answer:"/"Solution:" block that got trapped inside the STEM instead of
    being separated out — the failure mode where the entire worked derivation (and
    sometimes the answer key) ends up duplicated inside `stem` because the upstream
    extractor (LLM or regex) never drew the stem/answer boundary correctly.

    This is the stem-side counterpart to `separate_options_dict()`, which only ever
    cleaned `options`. Existing `marked_answer`/`explanation` values are preserved and
    take priority — a stem-trapped answer/explanation only fills in what's missing,
    it never overwrites a value that was already extracted correctly elsewhere.
    Returns (clean_stem, marked_answer, explanation).
    """
    if not stem or not isinstance(stem, str):
        return stem, marked_answer, explanation

    clean_stem, stem_ans, stem_exp = parse_trailing_meta(stem)
    if stem_ans and not marked_answer:
        marked_answer = stem_ans
    if stem_exp and not explanation:
        explanation = stem_exp
    # Only adopt the cleaned stem if something was actually found and stripped —
    # otherwise leave the original stem untouched (parse_trailing_meta can be
    # conservative-but-imperfect on stems that legitimately end in a colon).
    if stem_ans or stem_exp:
        stem = clean_stem
    return stem, marked_answer, explanation


def heal_assertion_reason_question(stem: str, options: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Detects and heals Assertion-Reason questions where the Assertion/Reason statements
    were accidentally trapped inside Option A, leaving the question stem truncated (e.g. just 'Assertion').
    """
    if not options or not isinstance(options, dict):
        return stem, options

    # Find Option A content
    key_a = None
    for k in ("A", "a", "1"):
        if k in options:
            key_a = k
            break
    if not key_a:
        return stem, options

    opt_a_text = str(options.get(key_a) or "").strip()
    if not opt_a_text:
        return stem, options

    # Check if opt_a contains Reason / കാരണം statement
    has_reason = bool(re.search(r'\b(?:reason|കാരണം)\s*(?:\([rR]\)|[\:\.\-]+|\s*$)', opt_a_text, re.IGNORECASE))
    stem_clean = (stem or "").strip()
    stem_is_short_assertion = bool(re.search(r'\b(?:assertion|പ്രസ്താവന)\b', stem_clean, re.IGNORECASE)) and len(re.sub(r'<[^>]+>', '', stem_clean).strip()) < 120

    if has_reason or stem_is_short_assertion:
        # Look for the embedded option A boundary inside opt_a_text
        # e.g. "(a) If both Assertion", "(a) Both A and R", "If both Assertion and Reason are true", "Both Assertion and Reason are true"
        split_match = re.search(
            r'(?:(?:^|\n|\s{2,}|\.\s+)\(?([aA1])\)?[\.\:\-]?\s+(?=(?:if\s+both|both|assertion|statement|[aA]\s+is\b)))|'
            r'(?:(?:^|\n)\s*(?=(?:if\s+both|both\s+(?:assertion|a\b|r\b)|assertion\s+is\s+true|[aA]\s+is\s+true)))',
            opt_a_text,
            re.IGNORECASE
        )
        if split_match:
            split_idx = split_match.start()
            statement_block = opt_a_text[:split_idx].strip()
            choice_text = opt_a_text[split_idx:].strip()
            # Strip leading option marker (e.g. "(a) ") from choice_text
            choice_text = re.sub(r'^\s*\(?[aA1]\)?[\.\:\-]?\s*', '', choice_text).strip()

            # Clean marks indicator from statement_block
            statement_block = re.sub(r'(?i)\s*\[\s*\d+(?:\.\d+)?(?:\s*/\s*\d+)?\s*(?:marks?|mark|m)\s*\]', '', statement_block)
            statement_block = re.sub(r'(?i)\s*\(\s*\d+(?:\.\d+)?(?:\s*/\s*\d+)?\s*(?:marks?|mark|m)\s*\)', '', statement_block)
            statement_block = statement_block.strip()

            if statement_block:
                if not stem_clean:
                    new_stem = f"Assertion (A): {statement_block}"
                elif re.match(r'^(?:assertion|പ്രസ്താവന)\s*[\:\.\-]?$', stem_clean, re.IGNORECASE):
                    new_stem = f"Assertion (A): {statement_block}"
                elif not re.search(r'\b(?:reason|കാരണം)\b', stem_clean, re.IGNORECASE):
                    new_stem = f"{stem_clean}\n{statement_block}"
                else:
                    new_stem = stem_clean

                new_options = dict(options)
                new_options[key_a] = choice_text
                return new_stem, new_options

    return stem, options


def clean_question_options_and_answer(q: Any) -> Any:
    """
    Standardizes a question dict or QuestionResult model in-place:
    - Separates options from answers and solutions
    - Strips any Answer:/Solution: block trapped inside the STEM itself
    - Validates & best-effort repairs broken LaTeX brace/delimiter structure across
      stem, options, and explanation; flags the question for review if repair was needed
    - Normalizes math in all option values
    - Auto-classifies question_type as 'mcq' or 'descriptive'
    """
    if q is None:
        return q

    is_dict = isinstance(q, dict)
    opts = (q.get("options") if is_dict else getattr(q, "options", None)) or {}
    stem = (q.get("stem") if is_dict else getattr(q, "stem", None)) or ""
    marked = (q.get("marked_correct_answer") or q.get("marked_answer")) if is_dict else (getattr(q, "marked_correct_answer", None) or getattr(q, "marked_answer", None))
    exp = (q.get("explanation") or q.get("answer_text")) if is_dict else (getattr(q, "explanation", None) or getattr(q, "answer_text", None))

    llm_q_type = (q.get("question_type") if is_dict else getattr(q, "question_type", None))

    # Stem-contamination pass FIRST, so an "Answer:"/"Solution:" block sitting in the
    # stem is stripped out and folded into marked/exp before option-cleaning runs.
    new_stem, marked, exp = clean_stem_contamination(stem, marked, exp)

    # Trailing option & answer stripping: extract dangling trailing options like '(D) 1.0 \times 10'
    # or answer choices from explanation/stem into opts before separate_options_dict runs.
    new_stem, opts, marked, exp = strip_and_extract_trailing_options_from_stem(new_stem, opts, marked, exp)

    # Check if sub_parts already exists on q. If invalid (e.g. empty part or marks-only part),
    # collapse it back into new_stem so diagrams and text are preserved in the stem!
    existing_sub = (q.get("sub_parts") if is_dict else getattr(q, "sub_parts", None))
    if existing_sub and not is_valid_subparts(existing_sub, stem=new_stem):
        parts_to_reintegrate = []
        for k in sorted(existing_sub.keys()):
            v_content = str(existing_sub[k] or "").strip()
            if v_content:
                if new_stem and re.search(r'\b(?:points?|between|and|from|to)\s+[a-zA-Z]\s+and\s*$', new_stem, re.IGNORECASE):
                    parts_to_reintegrate.append(f"{k.upper()}. {v_content}")
                else:
                    parts_to_reintegrate.append(v_content)
        if parts_to_reintegrate:
            rejoined = "\n".join(parts_to_reintegrate)
            new_stem = f"{new_stem}\n{rejoined}".strip() if new_stem else rejoined
        existing_sub = None
        if is_dict:
            q.pop("sub_parts", None)
        else:
            try:
                q.sub_parts = None
            except Exception:
                pass

    clean_opts, new_marked, new_exp = separate_options_dict(opts, marked, exp)
    new_stem, clean_opts = heal_assertion_reason_question(new_stem, clean_opts)

    # If all option values are identical to lines/steps in the explanation and there is no marked answer,
    # the options were extracted from the answer/solution block rather than being genuine MCQ options!
    if clean_opts and new_exp and not new_marked:
        all_in_exp = True
        for opt_val in clean_opts.values():
            val_clean = opt_val.strip()
            if val_clean and val_clean not in new_exp:
                all_in_exp = False
                break
        if all_in_exp:
            clean_opts = {}

    q_type, sub_parts = classify_question_type(clean_opts, new_exp, marked_answer=new_marked, stem=new_stem)
    if not sub_parts and existing_sub and is_valid_subparts(existing_sub, stem=new_stem):
        sub_parts = existing_sub
        q_type = "descriptive_subquestions"
        clean_opts = {}

    # LaTeX structural validation across every text field. Best-effort repair, but we
    # ALWAYS flag `needs_review` if any field had to be repaired — a mechanically fixed
    # brace count is not the same guarantee as "this LaTeX means what the source meant".
    needs_review = False
    latex_issues = []
    new_stem, stem_issue = validate_and_repair_latex(new_stem)
    if stem_issue:
        needs_review = True
        latex_issues.append("stem")
    for k in list(clean_opts.keys()):
        if isinstance(clean_opts[k], str):
            clean_opts[k], opt_issue = validate_and_repair_latex(clean_opts[k])
            if opt_issue:
                needs_review = True
                latex_issues.append(f"option_{k}")
    if new_exp:
        new_exp, exp_issue = validate_and_repair_latex(new_exp)
        if exp_issue:
            needs_review = True
            latex_issues.append("explanation")
    
    # If options are empty or incomplete (< 4), attempt to recover options from stem
    if not clean_opts or len(clean_opts) < 4:
        from ..graph.nodes import separate_stem_and_options
        rec_stem, rec_opts, rec_ans, rec_exp = separate_stem_and_options(new_stem)
        if rec_opts and len(rec_opts) >= 2:
            rec_type, rec_sub = classify_question_type(rec_opts, new_exp or rec_exp, marked_answer=new_marked or rec_ans, stem=rec_stem)
            if rec_type == "mcq" and len(rec_opts) > len(clean_opts):
                new_stem = rec_stem
                clean_opts = rec_opts
                if not new_marked and rec_ans:
                    new_marked = rec_ans
                if not new_exp and rec_exp:
                    new_exp = rec_exp
                q_type = "mcq"
                sub_parts = None
            elif rec_type == "descriptive_subquestions" and not sub_parts and rec_sub:
                sub_parts = rec_sub
                q_type = "descriptive_subquestions"
                clean_opts = {}
                if not new_marked and rec_ans:
                    new_marked = rec_ans
                if not new_exp and rec_exp:
                    new_exp = rec_exp

    # Adopt LLM-verified question_type if options are still empty and no MCQ detected
    if not clean_opts and llm_q_type and str(llm_q_type).strip().lower() in ("descriptive_single", "descriptive_subquestions", "descriptive"):
        q_type = str(llm_q_type).strip().lower()

    try:
        from .math_normalizer import normalize_math_in_text
        if new_stem and isinstance(new_stem, str):
            new_stem = normalize_math_in_text(new_stem)
        for k, v in list(clean_opts.items()):
            if isinstance(v, str):
                clean_opts[k] = normalize_math_in_text(v)
        if new_exp and isinstance(new_exp, str):
            new_exp = normalize_math_in_text(new_exp)
    except Exception:
        pass

    # If sub_parts not found from options, try extracting from stem
    if not sub_parts and new_stem and q_type != "mcq":
        extracted_sub = extract_subparts_from_text(new_stem)
        if extracted_sub and is_valid_subparts(extracted_sub, stem=new_stem):
            sub_parts = extracted_sub
            q_type = "descriptive_subquestions"

    # If sub_parts exists and is valid, extract only the introductory stem (strip sub-questions out of stem)
    # to prevent duplicating the question in both stem and sub_parts!
    if sub_parts and is_valid_subparts(sub_parts, stem=new_stem) and new_stem:
        first_key = sorted(sub_parts.keys())[0]
        fk_esc = re.escape(first_key)
        first_sub_pattern = re.compile(
            rf'(?:(?:^|\n|\s{{2,}}|\.\s+)\([{fk_esc}{fk_esc.upper()}]\)\s*)|'
            rf'(?:(?:^|\n|[ \t]{{2,}}|\.\s+)[{fk_esc}{fk_esc.upper()}][\.\)\:-]\s*)'
        )
        m_sub = first_sub_pattern.search(new_stem)
        if m_sub:
            intro = new_stem[:m_sub.start()].strip()
            if re.match(r"^(?:question\s*\d*|q\d*|ചോദ്യം\s*\d*|[\d\.\:\-\s])*$", intro, re.IGNORECASE):
                new_stem = ""
            else:
                new_stem = intro
    elif sub_parts and not is_valid_subparts(sub_parts):
        sub_parts = None

    if sub_parts and q_type != "mcq":
        q_type = "descriptive_subquestions"
        final_opts = {}
        new_marked = None
    elif q_type == "mcq" and clean_opts:
        final_opts = clean_opts
        sub_parts = None
    elif q_type.startswith("descriptive"):
        mcq_keys = set(str(k).upper() for k in clean_opts.keys())
        if len(clean_opts) >= 2 and mcq_keys.issubset({'A', 'B', 'C', 'D', 'E', '1', '2', '3', '4', '5', 'I', 'II', 'III', 'IV', 'V'}):
            q_type = "mcq"
            final_opts = clean_opts
            sub_parts = None
        else:
            q_type = "descriptive_single"
            final_opts = {}
            new_marked = None
            if clean_opts and not new_exp:
                new_exp = "\n".join(f"{k}. {v}" for k, v in clean_opts.items())
    else:
        final_opts = clean_opts

    if new_exp:
        new_exp = strip_subquestion_labels_from_answer(new_exp)

    if is_dict:
        q["stem"] = new_stem
        q["options"] = final_opts
        if new_marked:
            q["marked_answer"] = new_marked
            q["marked_correct_answer"] = new_marked
        elif q_type.startswith("descriptive"):
            q["marked_answer"] = None
            q["marked_correct_answer"] = None
            q["validated_answer"] = None
            q["is_valid"] = True
        if new_exp:
            q["explanation"] = new_exp
            q["answer_text"] = new_exp
        q["question_type"] = q_type
        if q_type == "mcq" or not sub_parts:
            q.pop("sub_parts", None)
        else:
            q["sub_parts"] = sub_parts

        # Surface anything that needed mechanical repair so the API/UI layer can route
        # this question to human review or an automatic rescan instead of shipping it
        # silently. Never clear a `needs_review` flag set by an earlier pipeline stage.
        if needs_review:
            q["needs_review"] = True
            q["_latex_issues"] = sorted(set(q.get("_latex_issues", []) + latex_issues))

        val_ans = q.get("validated_answer")
        cur_marked = q.get("marked_answer") or q.get("marked_correct_answer")
        if q_type.startswith("descriptive"):
            q["is_valid"] = True
            reasoning = q.get("reasoning") or ""
            if "differs from marked None" in reasoning or "disagreement" in reasoning.lower():
                q["reasoning"] = q.get("explanation") or "Descriptive question verified"
        elif val_ans and cur_marked:
            is_valid = bool(str(val_ans).strip().upper() == str(cur_marked).strip().upper())
            q["is_valid"] = is_valid
            reasoning = q.get("reasoning") or ""
            if is_valid and ("differs from marked None" in reasoning or "disagreement" in reasoning.lower()):
                q["reasoning"] = q.get("explanation") or f"Answer key verified and matches Option {val_ans}"
    else:
        q.stem = new_stem
        q.options = final_opts
        if new_marked:
            q.marked_answer = new_marked
            q.marked_correct_answer = new_marked
        elif q_type.startswith("descriptive"):
            q.marked_answer = None
            q.marked_correct_answer = None
            q.validated_answer = None
            q.is_valid = True
        if new_exp:
            q.explanation = new_exp
            q.answer_text = new_exp
        q.question_type = q_type
        if q_type == "mcq" or not sub_parts:
            q.sub_parts = None
        else:
            q.sub_parts = sub_parts

        if needs_review:
            try:
                q.needs_review = True
                q._latex_issues = sorted(set(getattr(q, "_latex_issues", []) or []) | set(latex_issues))
            except Exception:
                # Model may not declare these fields (e.g. a strict Pydantic model) —
                # don't let a missing attribute crash the whole cleaning pass.
                pass

        val_ans = getattr(q, "validated_answer", None)
        cur_marked = getattr(q, "marked_answer", None) or getattr(q, "marked_correct_answer", None)
        if q_type.startswith("descriptive"):
            q.is_valid = True
            reasoning = getattr(q, "reasoning", "") or ""
            if "differs from marked None" in reasoning or "disagreement" in reasoning.lower():
                q.reasoning = getattr(q, "explanation", None) or "Descriptive question verified"
        elif val_ans and cur_marked:
            is_valid = bool(str(val_ans).strip().upper() == str(cur_marked).strip().upper())
            q.is_valid = is_valid
            reasoning = getattr(q, "reasoning", "") or ""
            if is_valid and ("differs from marked None" in reasoning or "disagreement" in reasoning.lower()):
                q.reasoning = getattr(q, "explanation", None) or f"Answer key verified and matches Option {val_ans}"

    return q

def strip_subparts_text_from_stem(stem: str, sub_parts: Dict[str, str]) -> str:
    """
    Removes the text of subparts from the stem if they were duplicated by the LLM
    without their list markers (e.g., (a), (b)).
    """
    if not stem or not sub_parts:
        return stem
        
    clean_stem = stem
    for k, v in sub_parts.items():
        if len(v) > 10:
            import re
            v_pat = re.escape(v.strip())
            # Replace escaped whitespace with a flexible whitespace matcher
            v_pat = re.sub(r'\\\s+', r'\\s+', v_pat)
            clean_stem = re.sub(v_pat, '', clean_stem, flags=re.IGNORECASE)
            
    # Also strip any orphaned list markers left dangling at the end
    clean_stem = re.sub(r'(?:^|\n|\s+)\([a-zA-Z0-9]+\)[\.\:]?\s*$', '', clean_stem).strip()
    return clean_stem
