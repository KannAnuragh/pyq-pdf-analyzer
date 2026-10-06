import re
from typing import Optional

def is_instruction_or_header(text: str) -> bool:
    """Detects if a text snippet is exam instructions or document header rather than a question."""
    t = text.strip()[:200].lower()
    instruction_starters = (
        'all questions are',
        'each question carries',
        'the question paper contains',
        'the question paper consists',
        'general instructions',
        'instructions for candidate',
        'candidates are required',
        'time allowed',
        'maximum marks',
        'total marks',
        'read the following instructions',
        'do not use calculator',
        'use of calculator',
        'sections in the question paper',
        'there are 3 sections',
        'there are 4 sections',
    )
    return any(t.startswith(s) or f" {s}" in t for s in instruction_starters)

def is_equation_or_calculation(text: str) -> bool:
    """Detects if text is purely an equation or calculation remnant (no natural language question)."""
    clean = text.strip()
    # Remove math tags / LaTeX / markdown / numbers / symbols
    stripped = re.sub(r'\$[^$]*\$', '', clean)
    stripped = re.sub(r'<[^>]+>', '', stripped)
    stripped = re.sub(r'[\d\s\=\+\-\*\/\^\_\(\)\[\]\{\}\\\.\,\:\;\%\>\<\~\|\#]', '', stripped)
    # If fewer than 4 alphabetic / Indic characters remain, it's not a real question
    if len(stripped) < 4:
        return True
    return False

def parse_questions_from_text_improved(raw_text: str) -> list[dict]:
    """
    Robustly extracts question blocks from raw text.
    Prevents random texts, equations, decimal numbers, and instructions from being labelled as questions.
    """
    if not raw_text or not raw_text.strip():
        return []

    # 1. First, check if document uses explicit Question markers (e.g. 'Question 1:', 'Q. 1:', 'Question1:-', 'ചോദ്യം 1:')
    explicit_pat = re.compile(
        r'(?:^|\n)\s*[\*\_]*(?:Question\s*|Q\.?\s*|ചോദ്യം\s*|പ്രശ്നം\s*)[\*\_]*(\d{1,3})\s*[\*\_]*(?::\-?|[\.\):-])\s*[\*\_]*(?=\s+[^\=\+\-\*\/]|\s*$)',
        re.IGNORECASE
    )
    explicit_matches = list(explicit_pat.finditer(raw_text))

    if len(explicit_matches) >= 3:
        raw_matches = explicit_matches
        is_explicit = True
    else:
        is_explicit = False
        # General numeric question prefix at start of a line
        # Must be preceded by line start or newline
        # Punctuation must be dot '.', closing paren ')', or ':-'
        # Must NOT be part of decimal (e.g. 0.125 or 1.5)
        # Must NOT be preceded by an open paren (e.g. '(1)')
        # Must NOT be followed by math operators (=, +, -, *, /)
        num_pat = re.compile(
            r'(?:^|\n)([ \t]*)' # line start + indentation
            r'[\*\_]*(\d{1,3})[\*\_]*\s*' # 1-3 digits
            r'(?:[\.\)]|:-)\s*[\*\_]*' # dot, paren, or :-
            r'(?=[^\d\s\=\+\-\*\/]|\s+[^\=\+\-\*\/])', # followed by text, not digits or math operators
            re.IGNORECASE
        )
        
        bad_prev_words = {
            'section', 'part', 'class', 'unit', 'chapter', 'table', 'figure', 'fig',
            'date', 'time', 'marks', 'mark', 'page', 'code', 'no', 'sl', 'step', 'case'
        }
        raw_matches = []
        for m in num_pat.finditer(raw_text):
            start_pos = m.start()
            # Check the line or word immediately preceding
            line_before = raw_text[max(0, start_pos - 40):start_pos].strip().split('\n')[-1]
            last_word = line_before.split()[-1].rstrip('.:-').lower() if line_before.split() else ''
            if last_word in bad_prev_words:
                continue
            # If line_before ends with an open paren '(', skip (it's an inline '(1)')
            if line_before.endswith('('):
                continue
            raw_matches.append(m)

    # Filter candidate matches: valid numbers (1..300) and non-instructions
    candidates = []
    for m in raw_matches:
        q_num_str = m.group(1) if is_explicit else m.group(2)
        q_num = int(q_num_str)
        if q_num < 1 or q_num > 300:
            continue
        
        snippet = raw_text[m.end():m.end() + 200].strip()
        if is_instruction_or_header(snippet):
            continue
            
        candidates.append({
            'match': m,
            'q_num': q_num,
            'start': m.start(),
            'end': m.end()
        })

    if not candidates:
        return []

    # 2. Sequence Selection:
    # Find the dominant, valid sequence of question numbers.
    # Questions should be monotonically increasing (e.g. 1, 2, 3, 4...).
    # We allow section resets (e.g. resetting to 1 after reaching >= 5).
    valid_candidates = []
    expected_next = 1
    
    # Check if first candidate starts around 1 (or allow starting at min if reasonable)
    first_q_num = candidates[0]['q_num']
    if first_q_num <= 2:
        expected_next = first_q_num
    else:
        # Check if there is a 1 later in candidates
        has_one = any(c['q_num'] == 1 for c in candidates)
        if not has_one:
            expected_next = first_q_num

    for c in candidates:
        qn = c['q_num']
        # Allow exact match, small gap (skipped question up to +2), or section reset to 1
        if qn >= expected_next and qn <= expected_next + 3:
            valid_candidates.append(c)
            expected_next = qn + 1
        elif qn == 1 and expected_next > 5:
            # Section reset back to 1
            valid_candidates.append(c)
            expected_next = 2
        elif not valid_candidates and qn <= 3:
            # Starting point
            valid_candidates.append(c)
            expected_next = qn + 1

    if not valid_candidates:
        # Fallback to all candidates if sequence filtering was overly strict
        valid_candidates = candidates

    # 3. Slice text into blocks
    blocks = []
    for i, c in enumerate(valid_candidates):
        start_idx = c['end']
        end_idx = valid_candidates[i+1]['start'] if i + 1 < len(valid_candidates) else len(raw_text)
        content = raw_text[start_idx:end_idx].strip()
        
        # Validate that block content actually has substantive question text
        if content and not is_equation_or_calculation(content) and not is_instruction_or_header(content):
            blocks.append({
                'q_num': c['q_num'],
                'content': content
            })

    return blocks
