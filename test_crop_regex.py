import re
pat_prefix = r'(?i)(?:^|\n)\s*[\*\_]*(?:Question\s*(?:No\.?)?|Q\.?\s*(?:No\.?)?|Sl\.?\s*No\.?|Problem\s*|Prob\.?\s*|ചോദ്യം\s*|പ്രശ്നം\s*)?[\(\[]?\s*'
pat_suffix = r'\s*[\)\]]?\s*[\*\_]*(?::\-?|[\.\):-])(?:\s+|$|(?=[\(\[]))'
q_num = 5
next_q_pat = re.compile(f"{pat_prefix}{q_num}{pat_suffix}")

tests = [
    "5.",
    "5. Assertion",
    "\n5. ",
    "5)",
    "[5]",
    "(5)",
    "Answer: 5.",
    "(b) 5.0",
    "5.0",
    "5:"
]
for t in tests:
    print(f"{repr(t)} -> {bool(next_q_pat.search(t))}")
