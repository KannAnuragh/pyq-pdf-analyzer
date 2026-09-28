import re

content = """A steady current of 8 mA flows through a wire. The number of electrons passing through a cross-section of the wire in 10 s is: 
 
 
 
[1 mark] 
(a) 4.0 × 1016 
(b) 5.0 × 1017 
(c) 1.6 × 1016 
(d) 1.0 × 1017 
Answer: 
(b) 5.0 × 1017"""

opt_pattern = re.compile(
    r'(?:(?:^|\n|\s+)\(([A-Ha-h1-8])\)[\.\:\-]?\s*)|'
    r'(?:(?:^|\n|[ \t]+|\.\s+)([A-Ha-h1-8])[\.\)\:-]\s*)'
)
matches = []
for m in opt_pattern.finditer(content):
    letter = m.group(1) or m.group(2)
    start_idx = m.start(1) if m.group(1) else m.start(2)
    prefix = content[max(0, start_idx - 35):start_idx].lower()
    full_match = m.group(0).strip()
    has_paren = full_match.startswith("(")

    # 1. Reject candidate matches preceded by Assertion/Reason/Statement labels:
    if re.search(r'(?:assertion|reason|statement|പ്രസ്താവന|കാരണം)[\s\:\-\*\_\(]*$', prefix, re.IGNORECASE):
        continue
    # 2. Reject candidate matches preceded by prepositions / linkers / points:
    prev_word_match = re.search(r'([a-zA-Z]+)\s*(?:and\s*)?$', prefix)
    if not has_paren and prev_word_match:
        prev_word = prev_word_match.group(1).lower()
        reject_set = {'to', 'of', 'or', 'and', 'between', 'from',
                      'junction', 'terminal', 'point', 'points', 'cell', 'cells',
                      'charge', 'charges', 'plate', 'plates'}
        if len(prev_word) >= 2 and prev_word in reject_set:
            continue
    # Reject candidate matches preceded by digits (e.g. scientific unit "0.2 A.")
    if not has_paren and re.search(r'\d+(?:\.\d+)?\s*$', prefix):
        continue
    matches.append((letter.upper(), m))

print("Found matches:", [(l, m.group(0)) for l, m in matches])
