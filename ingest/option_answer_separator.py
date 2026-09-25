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

# Pattern matching Answer markers:
# Must be at the start of a line/string, OR separated by whitespace/tabs with an explicit ':' or '-' delimiter.
# Never match a bare word followed only by newline unless at the start of a line.
ANSWER_MARKER_PATTERN = re.compile(
    r"(?:(?:^|\n|\\n)\s*(?:\*\*\s*(?:Correct(?:\s*Option|\s*Answer)?|Answer|Ans(?:wer)?|ഉത്തരം|ശരിയായ\s*ഉത്തരം)\s*\*\*|\*?\*?(?:Correct(?:\s*Option|\s*Answer)?|Answer|Ans(?:wer)?|ഉത്തരം|ശരിയായ\s*ഉത്തരം)\*?\*?\s*(?:[\:\-]+\s*|(?=\n|\\n|$))))|"
    r"(?:(?:[ \t]+|\t)(?:\*\*\s*(?:Correct(?:\s*Option|\s*Answer)?|Answer|Ans(?:wer)?|ഉത്തരം|ശരിയായ\s*ഉത്തരം)\s*\*\*|\*?\*?(?:Correct(?:\s*Option|\s*Answer)?|Answer|Ans(?:wer)?|ഉത്തരം|ശരിയായ\s*ഉത്തരം)\*?\*?\s*[\:\-]+\s*))",
    re.IGNORECASE
)

# Pattern matching Solution / Explanation markers:
# A true solution marker is a SECTION HEADER.
# It can only be:
# 1. At line start (or preceded by \n) followed by delimiter (:, -, .-) or on its own line before newline:
#    e.g. "\nExplanation:", "\nSolution\n", "\n**Explanation:**", "\nവിശദീകരണം:"
# 2. OR separated by 2+ spaces or tabs with explicit delimiter (:, -, .-)
# NEVER allow a single space in the middle of a sentence without delimiter, because common exam words like
# "... Reason is correct explanation of Assertion" will falsely match and destroy options and explanations!
SOLUTION_MARKER_PATTERN = re.compile(
    r"(?:(?:^|\n|\\n)\s*(?:\*\*\s*(?:Solution|Explanation|Description|Sol|Exp|Hint|Hints|Reasoning|വിശദീകരണം|പരിഹാരം|വിവരണം|സൂചന)\s*\*\*|\*?\*?(?:Solution|Explanation|Description|Sol|Exp|Hint|Hints|Reasoning|വിശദീകരണം|പരിഹാരം|വിവരണം|സൂചന)\*?\*?\s*(?:[\:\.\-]+\s*|(?=\n|\\n|$))))|"
    r"(?:(?:[ \t]{2,}|\t)(?:\*\*\s*(?:Solution|Explanation|Description|Sol|Exp|Hint|Hints|Reasoning|വിശദീകരണം|പരിഹാരം|വിവരണം|സൂചന)\s*\*\*|\*?\*?(?:Solution|Explanation|Description|Sol|Exp|Hint|Hints|Reasoning|വിശദീകരണം|പരിഹാരം|വിവരണം|സൂചന)\*?\*?\s*[\:\.\-]+\s*))",
    re.IGNORECASE
)

# Pattern for option letter at start of answer tail (supports e.g. (b), [b], b, Option B, etc., followed by math/text)
ANSWER_LETTER_PATTERN = re.compile(
    r"^(?:Option\s*[:\-]*\s*)?(?:\(\s*([A-Ha-h])\s*\)|\[\s*([A-Ha-h])\s*\]|([A-Ha-h]))(?=\s|[\.\:\-\$\\\\]|$)",
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
            has_multi_sub = len(re.findall(r'(?:^|\n|\s)\(([a-d])\)[\s\:\-\.]+', ans_part, re.IGNORECASE)) >= 2
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
            has_multi_sub = len(re.findall(r'(?:^|\n|\s)\(([a-d])\)[\s\:\-\.]+', ans_part, re.IGNORECASE)) >= 2
            if not has_multi_sub:
                l_match = ANSWER_LETTER_PATTERN.match(ans_part)
                if l_match:
                    marked_answer = (l_match.group(1) or l_match.group(2) or l_match.group(3)).upper()
            else:
                explanation = f"{explanation}\n{ans_part}" if explanation else ans_part
    elif ans_m:
        tail = clean_text[ans_m.end():].strip()
        clean_text = clean_text[:ans_m.start()].strip()
        has_multi_sub = len(re.findall(r'(?:^|\n|\s)\(([a-d])\)[\s\:\-\.]+', tail, re.IGNORECASE)) >= 2
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
            r"(?:Option\s*[:\-]*\s*\(?([A-D])\)?\s*is\s*correct|\(?([A-D])\)?\s*is\s*(?:the\s*)?correct\s*(?:option|answer))",
            explanation,
            re.IGNORECASE
        )
        if ans_in_sol:
            marked_answer = (ans_in_sol.group(1) or ans_in_sol.group(2)).upper()
        elif explanation:
            leading_ans = re.match(r"^\(?([A-D])\)?(?=[^a-zA-Z0-9]|$)", explanation, re.IGNORECASE)
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
        if marked_answer:
            explanation = re.sub(rf"^\(?{re.escape(marked_answer)}\)?[\s\.\:\-]+", "", explanation, flags=re.IGNORECASE).strip()
            if explanation.strip("() \n\t.").upper() == marked_answer.upper():
                explanation = None

    if not marked_answer:
        # Check if text ends with a naked answer like [A] or (C) on its own line
        naked_pat = re.compile(r'(?:^|\n)\s*[\[\(]([A-D])[\]\)]\s*$', re.IGNORECASE)
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
            elif len(str(current_k)) == 1 and str(current_k).isalpha():
                next_ord = ord(str(current_k).upper()) + 1
                cand_char = chr(next_ord)
                cand_chars = [cand_char, cand_char.lower()] if next_ord <= ord('H') else []
            else:
                cand_chars = []

            if not cand_chars:
                out[current_k] = current_text
                break

            escaped_cands = [re.escape(c) for c in cand_chars]
            cand_group = "|".join(escaped_cands)

            # Look for:
            # 1. Parenthesized/bracketed: (B), [B], (b), [b]
            # 2. Bare letter: B., B), B:- (preceded by space/newline, followed by space)
            pat = re.compile(
                rf'(?:^|\n|[ \t]+)(?:\(([{cand_group}])\)|\[([{cand_group}])\]|([{cand_group}])[\.\)\:-])\s+'
            )
            m = pat.search(current_text)
            if m:
                split_idx = m.start()
                first_part = current_text[:split_idx].strip()
                matched_char = m.group(1) or m.group(2) or m.group(3)
                next_k = matched_char.upper() if current_k.isupper() else matched_char
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
        pat = rf"^\s*(?:\([\s]*[{k_esc}{k_esc.lower()}][\s]*\)[\.\:\-]?|\[[\s]*[{k_esc}{k_esc.lower()}][\s]*\][\.\:\-]?|[{k_esc}{k_esc.lower()}][\.\)\:\-])\s+"
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

        # Separate trailing answer or explanation from option body
        opt_body, ans, exp = parse_trailing_meta(v_clean)
        clean_opts[k] = opt_body

        if ans and not extracted_ans:
            extracted_ans = ans
        if exp and not extracted_exp:
            extracted_exp = exp
    # Override extracted answer if the explanation explicitly starts with an option letter (e.g., "(C) E != 0" or "Answer: A")
    if extracted_exp:
        pat1 = r"^\s*(?:Answer|Ans|Solution|Sol|Correct\s*Option|Option|Key)[\s\:\-\.]*[\(\[]?\s*([A-D])\s*[\)\]]?(?:[\s\.\:\-]+|is\s*correct\b)?"
        pat2 = r"^\s*(?:[\(\[]\s*([A-D])\s*[\)\]]|([A-D])\s*[\.\:\-\)]+)(?=\s|$)"
        
        m1 = re.match(pat1, extracted_exp, re.IGNORECASE)
        m2 = re.match(pat2, extracted_exp, re.IGNORECASE)
        
        explicit_match = m1 or m2
        if explicit_match:
            candidate = (explicit_match.group(1) if m1 else (m2.group(1) or m2.group(2))).upper()
            # Only override if this candidate actually exists in the options
            if any(k.upper() == candidate for k in clean_opts.keys()):
                extracted_ans = candidate
                # We no longer strip the prefix so that "Answer: C" remains visible in the explanation block.

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


def is_valid_subparts(sub_parts: Optional[Dict[str, Any]]) -> bool:
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
       or explicit marks allocation. Pure answer steps are strictly rejected.
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
    answer_indicators = (
        'higher', 'lower', 'increases', 'decreases', 'constant', 'remains same',
        'zero', 'positive', 'negative', 'forward bias', 'reverse bias',
        'answer:', 'ans:', 'solution:', 'sol:', 'hence', 'therefore', 'we get'
    )

    has_question_feature = False
    valid_count = 0

    for k, v in sub_parts.items():
        v_str = str(v or "").strip()
        if not v_str:
            return False
        # Remove html tags like <img .../> and <div ...> to inspect actual question text
        text_only = re.sub(r'<[^>]+>', '', v_str).strip()
        if not text_only or marks_only_pat.match(text_only):
            return False

        t_low = text_only.lower()

        # Reject pure answer derivations / comparative states without question directives
        is_answer_word = any(ans in t_low for ans in answer_indicators)
        has_multi_eq = "=" in text_only and len(text_only.split("=")) >= 2

        # Check for directive question features:
        has_qmark = "?" in text_only
        has_marks = bool(re.search(r'\[\s*\d+\s*marks?\s*\]|\(\s*\d+\s*marks?\s*\)|\b\d+\s*marks?\b', text_only, re.IGNORECASE))
        has_directive = any(d in t_low for d in eng_directives) or any(m in t_low for m in mal_directives)

        if has_qmark or has_marks or has_directive:
            has_question_feature = True
        elif is_answer_word or has_multi_eq:
            # Answer step with no question directive is NEVER a subquestion!
            return False

        # Must have some alphabetical / Indic characters
        if len(re.findall(r'[a-zA-Z\u0D00-\u0D7F]', text_only)) >= 3:
            valid_count += 1

    return (valid_count >= 2) and has_question_feature


def classify_question_type(options: Dict[str, Any], explanation: Optional[str] = None, marked_answer: Optional[str] = None) -> Tuple[str, Optional[Dict[str, str]]]:
    """
    Determines if a question is MCQ or descriptive (single or with subquestions).
    
    MCQ: options are standard choices (A, B, C, D) with short answer choices (values, formulas).
    Descriptive:
      - Has sub-questions: options are subparts like (a), (b) asking to explain, calculate, derive, define
      - Or contains question marks / marks indicators [1 mark]
      - Or explanation contains subpart solutions like (a)... (b)...
      - Or options are empty / single open-ended question.
    """
    if not options or len(options) < 2:
        return "descriptive_single", None

    keys_upper = sorted([k.upper() for k in options.keys()])
    sub_parts = {k.lower(): v for k, v in options.items()}

    # Helper to return descriptive_subquestions ONLY if subparts are genuinely valid
    def _subq_or_single(parts: Dict[str, str]) -> Tuple[str, Optional[Dict[str, str]]]:
        filtered_parts = {
            k: v for k, v in parts.items()
            if str(v).strip().lower() not in (
                "none of the above", "all of the above", "none of these", "all of these", "none", "all"
            )
        }
        target_parts = filtered_parts if len(filtered_parts) >= 2 else parts
        if is_valid_subparts(target_parts):
            return "descriptive_subquestions", target_parts
        return "descriptive_single", None

    # Check if option texts look like genuine directive sub-questions (e.g. "Calculate...", "State...")
    mal_directives = (
        'എഴുതുക', 'വിശദീകരിക്കുക', 'കണക്കാക്കുക', 'വരയ്ക്കുക', 'നിർവചിക്കുക',
        'കണ്ടെത്തുക', 'തെളിയിക്കുക', 'വ്യത്യാസ', 'രൂപീകരിക്കുക', 'പ്രസ്താവിക്കുക',
        'താരതമ്യം', 'പട്ടികപ്പെടുത്തുക', 'സൂചിപ്പിക്കുക'
    )
    eng_directives = (
        'draw', 'calculate', 'find', 'explain', 'state', 'prove',
        'derive', 'show', 'determine', 'define', 'describe',
        'distinguish', 'differentiate', 'obtain', 'evaluate', 'name',
        'what', 'why', 'how', 'which'
    )

    sub_q_count = 0
    for v in options.values():
        if isinstance(v, str):
            v_low = v.strip().lower()
            if any(m in v_low for m in mal_directives):
                sub_q_count += 1
            elif any(v_low.startswith(e) for e in eng_directives):
                sub_q_count += 1

    if sub_q_count >= 1:
        return _subq_or_single(sub_parts)

    # 1. Inspect explanation for numbered answer steps (e.g. "(a) ... (b) ..." or "1. ... 2. ... 3. ...")
    # These are ANSWER STEPS, NOT subquestions! Return descriptive_single with no subparts.
    if explanation and isinstance(explanation, str):
        exp_subq_matches = set(re.findall(r'(?:^|\n|\s)\(([a-d])\)[\s\:\-\.]+', explanation, re.IGNORECASE))
        if len(exp_subq_matches) >= 2:
            return "descriptive_single", None
        exp_numbered_items = len(re.findall(r'(?:^|\n)\s*(?:[1-9]\.|\([1-9]\)|\([ivx]+\)[\.\:]?)\s+', explanation, re.IGNORECASE))
        if exp_numbered_items >= 2:
            return "descriptive_single", None

    # 2. Check if option values contain explicit marks indicators [1 mark], [2 marks]
    marks_pat = re.compile(r'\[\s*\d+\s*marks?\s*\]|\(\s*\d+\s*marks?\s*\)|\b\d+\s*marks?\b', re.IGNORECASE)
    has_marks = any(isinstance(v, str) and marks_pat.search(v) for v in options.values())
    if has_marks:
        if is_valid_subparts(sub_parts):
            return _subq_or_single(sub_parts)
        return "descriptive_single", None

    # 3. Check if options contain synthetic padding (e.g. "None of the above") on top of answer steps
    # These are answers mistakenly parsed as options -> descriptive_single, no subparts!
    non_dummy = {
        k: v for k, v in options.items()
        if str(v).strip().lower() not in (
            "none of the above", "all of the above", "none of these", "all of these", "none", "all"
        )
    }
    if len(non_dummy) < len(options) and len(non_dummy) >= 2:
        has_states = any(str(v).strip().lower() in ("higher", "lower", "increase", "decrease", "forward bias", "reverse bias") for v in non_dummy.values())
        has_multi_eq = any("=" in str(v) and len(str(v).split("=")) >= 2 for v in non_dummy.values())
        if has_states or has_multi_eq:
            return "descriptive_single", None

    # 4. Check if options mix comparative states with multi-variable calculations
    has_comparative_state = any(str(v).strip().lower() in ("higher", "lower", "increases", "decreases") for v in options.values())
    has_multi_eq = any("=" in str(v) and len(str(v).split("=")) >= 2 for v in options.values())
    if has_comparative_state and has_multi_eq:
        return "descriptive_single", None

    # If marked answer is provided and matches an option key, it is an MCQ!
    if marked_answer and str(marked_answer).strip().upper() in keys_upper:
        return "mcq", None

    # ASSERTION-REASON DETECTION: If any option text mentions "both Assertion and Reason",
    # "Assertion is true", or similar truth-value evaluations, force MCQ classification.
    assertion_reason_signals = 0
    for v in options.values():
        if isinstance(v, str):
            v_low = v.strip().lower()
            if re.search(r'(?:both\s+)?assertion\s+and\s+reason', v_low, re.IGNORECASE):
                assertion_reason_signals += 1
            elif re.search(r'assertion\s+is\s+(?:true|false|correct)', v_low, re.IGNORECASE):
                assertion_reason_signals += 1
    if assertion_reason_signals >= 2:
        return "mcq", None

    valid_mcq_sets = (
        ["A", "B", "C", "D"],
        ["A", "B", "C"],
        ["A", "B", "C", "D", "E"],
        ["A", "B"],
        ["1", "2", "3", "4"],
        ["1", "2", "3"],
        ["1", "2"]
    )
    if keys_upper not in valid_mcq_sets:
        return _subq_or_single(sub_parts)

    # 3. Check if option texts look like genuine directive sub-questions
    # (Must be imperative commands instructing students to perform an exercise)
    mal_directives = (
        'എഴുതുക', 'വിശദീകരിക്കുക', 'കണക്കാക്കുക', 'വരയ്ക്കുക', 'നിർവചിക്കുക',
        'കണ്ടെത്തുക', 'തെളിയിക്കുക', 'വ്യത്യാസ', 'രൂപീകരിക്കുക', 'പ്രസ്താവിക്കുക',
        'താരതമ്യം', 'പട്ടികപ്പെടുത്തുക', 'സൂചിപ്പിക്കുക'
    )
    eng_directives = (
        'draw', 'calculate', 'find', 'explain', 'state', 'prove',
        'derive', 'show', 'determine', 'define', 'describe',
        'distinguish', 'differentiate', 'obtain', 'evaluate', 'name',
        'what', 'why', 'how', 'which'
    )

    sub_q_count = 0
    for v in options.values():
        if isinstance(v, str):
            v_low = v.strip().lower()
            if any(m in v_low for m in mal_directives):
                sub_q_count += 1
            elif any(re.search(r'\b' + re.escape(e) + r'\b', v_low) for e in eng_directives):
                sub_q_count += 1

    # Require at least 1 choice to have directive verbs before turning standard choices into subquestions
    if sub_q_count >= 1:
        return _subq_or_single(sub_parts)

    # If options contain images/diagrams and keys are standard choices, it is definitively an MCQ!
    if any(isinstance(v, str) and ('<img' in v or '<svg' in v) for v in options.values()):
        return "mcq", None

    # FORMULA-ONLY OPTIONS: If ALL option values are purely mathematical expressions
    # (LaTeX, fractions, numbers with units, single symbols), force MCQ classification.
    # These are answer choices, not sub-questions.
    formula_opt_count = 0
    for v in options.values():
        if isinstance(v, str):
            # Strip LaTeX, HTML, numbers, math symbols, units to see if any directive text remains
            v_stripped = re.sub(r'\$[^$]*\$', '', v)
            v_stripped = re.sub(r'<[^>]+>', '', v_stripped)
            v_stripped = re.sub(r'\\(?:frac|sqrt|times|div|pm|cdot|hat|vec|overline)\b', '', v_stripped)
            v_stripped = re.sub(r'[\d\s\=\+\-\*\/\^\_{\}\(\)\[\]\\\.,\:\;\%×÷±√πσεϕθμτ°²³⁻¹⁰]', '', v_stripped)
            v_stripped = re.sub(r'\b[a-zA-Z]{1,3}\b', '', v_stripped)  # Single short variables/units
            if len(v_stripped.strip()) < 3:
                formula_opt_count += 1
    if formula_opt_count == len(options) and len(options) >= 2:
        return "mcq", None

    # 4. Check if option texts are unreasonably long for MCQ choices (> 350 chars avg of actual text)
    # Strip HTML tags (e.g. <img .../>) so diagram choices are never penalized or converted into sub-questions!
    clean_lens = [len(re.sub(r'<[^>]+>', '', str(v)).strip()) for v in options.values()]
    avg_len = sum(clean_lens) / max(1, len(options))
    if avg_len > 350:
        return _subq_or_single(sub_parts)

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


def extract_subparts_from_text(text: str) -> Optional[Dict[str, str]]:
    """
    Extracts subquestions (e.g. (a), (b), (c) or (i), (ii)) from text
    when they are not already in sub_parts or options.
    Returns a dict mapping part key ('a', 'b', etc.) to the subquestion text,
    or None if no valid subquestion sequence is found.
    Answers and solutions are NEVER scanned or extracted as subquestions.
    """
    if not text or not isinstance(text, str):
        return None

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
        r'(?:(?:^|\n|\s{2,}|\.\s+)\(([a-hA-H1-8]|[ivxlcdmIVXLCDM]+)\)\s*)|'
        r'(?:(?:^|\n|[ \t]{2,}|\.\s+)([a-hA-H]|[ivxlcdmIVXLCDM]+)[\.\)\:-]\s*)'
    )
    raw_matches = list(subq_pattern.finditer(text))
    if len(raw_matches) < 2:
        return None

    valid_matches = []
    for m in raw_matches:
        k = (m.group(1) or m.group(2)).lower()
        start_idx = m.start(1) if m.group(1) else m.start(2)
        prefix = text[max(0, start_idx - 35):start_idx].lower()
        has_paren = bool(m.group(1))

        # Reject candidate matches preceded by Assertion/Reason/Statement labels:
        if re.search(r'(?:assertion|reason|statement|പ്രസ്താവന|കാരണം)[\s\:\-\*\_\(]*$', prefix, re.IGNORECASE):
            continue
        # Reject candidate matches preceded by prepositions / linkers / points:
        # ONLY reject if the preceding word is a full word (2+ chars)
        prev_word_m = re.search(r'([a-zA-Z]+)\s*(?:and\s*)?$', prefix)
        if prev_word_m:
            pw = prev_word_m.group(1).lower()
            reject_set = {'both', 'between', 'from', 'to', 'and', 'or', 'is', 'are', 'was', 'were',
                         'if', 'whether', 'that', 'when', 'where', 'of', 'in', 'for', 'with',
                         'junction', 'terminal', 'point', 'points', 'cell', 'cells',
                         'charge', 'charges', 'plate', 'plates'}
            if len(pw) >= 2 and pw in reject_set:
                continue
        # Reject candidate matches preceded by digits (e.g. scientific unit "0.2 A.")
        if not has_paren and re.search(r'\d+(?:\.\d+)?\s*$', prefix):
            continue
        valid_matches.append((k, m))

    if len(valid_matches) < 2:
        return None

    keys = [item[0] for item in valid_matches]
    first_key = keys[0]
    if first_key not in ('a', '1', 'i'):
        return None

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
        return None

    return subparts


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

    # Check if sub_parts already exists on q. If invalid (e.g. empty part or marks-only part),
    # collapse it back into new_stem so diagrams and text are preserved in the stem!
    existing_sub = (q.get("sub_parts") if is_dict else getattr(q, "sub_parts", None))
    if existing_sub and not is_valid_subparts(existing_sub):
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
    q_type, sub_parts = classify_question_type(clean_opts, new_exp, marked_answer=new_marked)
    if not sub_parts and existing_sub and is_valid_subparts(existing_sub):
        sub_parts = existing_sub

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
    
    # Adopt LLM-verified question_type if present
    if llm_q_type and str(llm_q_type).strip().lower() in ("mcq", "descriptive_single", "descriptive_subquestions", "descriptive"):
        q_type = str(llm_q_type).strip().lower()
        if q_type == "mcq" and not clean_opts:
            from ..graph.nodes import separate_stem_and_options
            rec_stem, rec_opts, rec_ans, rec_exp = separate_stem_and_options(new_stem)
            if rec_opts and len(rec_opts) >= 2:
                new_stem = rec_stem
                clean_opts = rec_opts
                if not new_marked and rec_ans:
                    new_marked = rec_ans
                if not new_exp and rec_exp:
                    new_exp = rec_exp

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
        if extracted_sub and is_valid_subparts(extracted_sub):
            sub_parts = extracted_sub
            q_type = "descriptive_subquestions"

    # If sub_parts exists and is valid, extract only the introductory stem (strip sub-questions out of stem)
    # to prevent duplicating the question in both stem and sub_parts!
    if sub_parts and is_valid_subparts(sub_parts) and new_stem:
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