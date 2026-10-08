import re
pat_prefix = r'(?i)(?:^|\n)\s*[\*\_]*(?:Question\s*(?:No\.?)?|Q\.?\s*(?:No\.?)?|Sl\.?\s*No\.?|Problem\s*|Prob\.?\s*|ചോദ്യം\s*|പ്രശ്നം\s*)?[\(\[]?\s*'
pat_suffix = r'\s*[\)\]]?\s*[\*\_]*(?::\-?|[\.\):-])(?:\s+|$|(?=[\(\[]))'
q_start_pat = re.compile(f"{pat_prefix}4{pat_suffix}")
print("Match Q4:", q_start_pat.search('4. A 300 Ω resistor and a capacitor of (\n25'))
next_q_pat = re.compile(f"{pat_prefix}5{pat_suffix}")
print("Match Q5:", next_q_pat.search('5. Assertion (A): When capacitive reactance is smaller than the inductive reactance'))
