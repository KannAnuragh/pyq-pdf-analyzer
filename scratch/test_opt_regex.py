import re

text = """5. Assertion (A): Work done in moving a charge around a closed path, in an electric field is always zero.  
Reason (R): Electrostatic force is a conservative force.      
 
[1 mark] 
(a) If both Assertion and Reason are true and Reason is correct explanation of Assertion. 
(b) If both Assertion and Reason are true but Reason is not the correct explanation of Assertion. 
(c) If Assertion is true but Reason is false.  
(d) If both Assertion and Reason are false. 
Answer: 
(a) If both Assertion and Reason are true and Reason is correct explanation of Assertion."""

opt_pattern = re.compile(r'(?:^|\n|\s+)(?:\()?([A-Da-d])(?:\)|\.|:-)\s*')
matches = []
for m in opt_pattern.finditer(text):
    prefix = text[:m.start(1)].lower()
    if re.search(r'(?:assertion|reason)\s*\($', prefix):
        continue
    matches.append(m)
    print(f"Accepted Match: '{m.group(0)}' (Group 1: {m.group(1)})")

