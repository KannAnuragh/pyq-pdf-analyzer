import re

def test():
    pat_prefix = r'(?i)(?:^|\n)\s*[\*\_]*(?:Question\s*(?:No\.?)?|Q\.?\s*(?:No\.?)?|Sl\.?\s*No\.?|Problem\s*|Prob\.?\s*|ചോദ്യം\s*|പ്രശ്നം\s*)?[\(\[]?\s*'
    pat_suffix = r'\s*[\)\]]?\s*[\*\_]*(?::\-?|[\.\):-])(?:\s+|$|(?=[\(\[]))'
    
    next_q_pat = re.compile(f"{pat_prefix}2{pat_suffix}")
    
    test_strings = [
        "2.5 kg",
        "2. ",
        "b) 2.",
        "Answer: 2.",
        "2. (a) 5",
        "2",
        "Ans: 2",
        "answer of subquestion a is 2."
    ]
    for s in test_strings:
        print(f"'{s}' -> {bool(next_q_pat.search(s))}")

test()
