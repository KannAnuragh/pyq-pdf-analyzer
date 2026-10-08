import re
pat_prefix = r'(?i)(?:^|\n)\s*[\*\_]*(?:Question\s*(?:No\.?)?|Q\.?\s*(?:No\.?)?|Sl\.?\s*No\.?|Problem\s*|Prob\.?\s*|ചോദ്യം\s*|പ്രശ്നം\s*)?[\(\[]?\s*'
pat_suffix = r'\s*[\)\]]?\s*[\*\_]*(?::\-?|[\.\):-])(?:\s+|$|(?=[\(\[]))'
q_num = 5
next_q_pat = re.compile(f"{pat_prefix}{q_num}{pat_suffix}")
print("Matches '(b) 0.5 A':", next_q_pat.search('(b) 0.5 A'))
