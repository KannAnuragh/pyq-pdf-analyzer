import re

def parse_trailing_meta(text):
    return text, None, None

def separate_stem_and_options(content: str):
    if not content:
        return "", {}, None, None

    # Option detection pattern
    opt_pattern = re.compile(
        r'(?:(?:^|\n|\s+)\(([A-Ea-e1-4])\)[\.\:\-]?\s*)|'
        r'(?:(?:^|\n|[ \t]+|\.\s+)([A-Ea-e])[\.\)\:-]\s+)'
    )
    matches = []
    for m in opt_pattern.finditer(content):
        letter = m.group(1) or m.group(2)
        start_idx = m.start(1) if m.group(1) else m.start(2)
        prefix = content[max(0, start_idx - 35):start_idx].lower()
        full_match = m.group(0).strip()
        has_paren = full_match.startswith("(")

        post_match = content[m.end():m.end() + 10].strip()
        if not has_paren and re.match(r'^[\=\+\-\*\/\^\_\±\×\÷\√]', post_match):
            continue
        elif has_paren and post_match.startswith('='):
            continue

        if re.search(r'(?:assertion|reason|statement|പ്രസ്താവന|കാരണം)[\s\:\-\*\_\(]*$', prefix, re.IGNORECASE):
            continue
        
        prev_word_match = re.search(r'([a-zA-Z]+)\s*(?:and\s*)?$', prefix)
        if prev_word_match:
            prev_word = prev_word_match.group(1).lower()
            if not has_paren:
                reject_words = {'both', 'between', 'from', 'and', 'or', 'is', 'are', 'was', 'were',
                               'if', 'whether', 'that', 'when', 'where', 'of', 'in', 'for', 'with',
                               'junction', 'terminal', 'point', 'points', 'cell', 'cells',
                               'charge', 'charges', 'plate', 'plates', 'to', 'number', 'mass',
                               'each', 'fragment', 'fragments', 'side', 'radius', 'amplitude'}
                if len(prev_word) >= 2 and prev_word in reject_words:
                    continue
            else:
                structural_reject = {'point', 'points', 'junction', 'terminal', 'cell', 'cells', 'number', 'mass'}
                if prev_word in structural_reject:
                    continue
        
        if not has_paren and full_match.endswith('.') and re.search(r'\d+(?:\.\d+)?\s*$', prefix):
            continue
        
        post_snippet = content[m.end():m.end() + 25].strip().lower()
        if re.match(r'^(?:[\:\.\-]+\s*)?(?:assertion|reason|statement|പ്രസ്താവന|കാരണം)\b', post_snippet, re.IGNORECASE):
            continue
        print(f"MATCH: {letter}, {m.group(0)}"); matches.append((letter, m))
    
    best_sequence = []
    for i, m_info in enumerate(matches):
        letter = m_info[0].upper()
        if letter != 'A' and letter != '1':
            continue
        curr_seq = [m_info]
        curr_letter = letter
        for next_m in matches[i+1:]:
            next_letter = next_m[0].upper()
            
            if next_letter > curr_letter:
                distance = next_m[1].start() - curr_seq[-1][1].end()
                if distance > 1500:
                    continue
                letter_gap = ord(next_letter) - ord(curr_letter)
                if letter_gap <= 2: 
                    curr_seq.append(next_m)
                    curr_letter = next_letter
        
        if len(curr_seq) >= 2 and len(curr_seq) > len(best_sequence):
            best_sequence = curr_seq

    stem = content.strip()
    options = {}
    marked_answer = None
    explanation = None

    if len(best_sequence) >= 2:
        stem = content[:best_sequence[0][1].start()].strip()
        for idx, m_info in enumerate(best_sequence):
            letter = m_info[0]
            m = m_info[1]
            start_pos = m.end()
            end_pos = best_sequence[idx+1][1].start() if idx < len(best_sequence) - 1 else len(content)
            opt_text = content[start_pos:end_pos].strip()
            
            options[letter] = opt_text

    return stem, options, marked_answer, explanation

content = """12. In which year was the Madras Corporation established?
A) 1588
B) 1688
C) 1788
D) 1888
Answer: B"""
print("Running test...")
print(separate_stem_and_options(content))
