import re

text = '''FINAL ANSWER KEY
Question Paper Code: 144/2026/OL
Question1:- ഇൻലാ ് വെസൽആക് ് 2021 കാര നിയമളെഎ ചാപ് കളായിതിരിിരി ?
        A:- X
        B:- XV
        C:- X X
        D:- XVIII
        Correct Answer:- Option-D
Question2:-
ഇൻലാ ് വെസൽആക് ് 2021 കാരംഉൾനാടൻജലാശയളിൽ 1.5 മീ ർ   തിരമാല )ടെഉയരം
ഉ   *ലളെഏത് സോണിൽഉൾെ ./0ിയിരി ?
        A:- സോൺ 1
        B:- സോൺ 2
        C:- സോൺ 3
        D:- ഇവയൊ മ2
        Correct Answer:- Option-A
'''

q_prefix_pattern = r'(?:Question\s*|Q\s*)?\d+(?:[\.\)-]|:-)'
blocks = re.split(fr'\n(?={q_prefix_pattern})|^(?={q_prefix_pattern})', text, flags=re.MULTILINE | re.IGNORECASE)

for block in blocks:
    block = block.strip()
    if not block: continue
    q_match = re.match(r'^(?:Question\s*|Q\s*)?(\d+)(?:[\.\)-]|:-)\s*(.*)', block, flags=re.DOTALL | re.IGNORECASE)
    if not q_match: continue
    q_num = int(q_match.group(1))
    content = q_match.group(2)
    options_pattern = re.compile(r'(?:^|\s)(?:\()?([A-D])(?:\)|\.|:-)\s*', re.IGNORECASE | re.MULTILINE)
    parts = options_pattern.split(content)
    stem = parts[0].strip()
    options = {}
    marked_answer = None
    for i in range(1, len(parts) - 1, 2):
        opt_letter = parts[i].upper()
        opt_text = parts[i+1].strip()
        ans_match = re.search(r'(?:Ans|Answer|Correct)\s*[:\-]?\s*([A-D])', opt_text, re.IGNORECASE)
        if ans_match:
            marked_answer = ans_match.group(1).upper()
            opt_text = opt_text[:ans_match.start()].strip()
        options[opt_letter] = opt_text
    
    if not marked_answer:
        ans_match = re.search(r'(?:Ans|Answer|Correct)\s*[:\-]?\s*([A-D])', stem, re.IGNORECASE)
        if ans_match:
            marked_answer = ans_match.group(1).upper()
            stem = stem[:ans_match.start()].strip()
            
    print(f"Q{q_num}: Marked={marked_answer}")
