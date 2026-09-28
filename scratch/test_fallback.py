import sys
sys.path.append("/home/anuragh/Trogon/qpaper scanner/pdf_qa_agent")
from graph.nodes import separate_stem_and_options

flat = """A steady current of 8 mA flows through a wire. The number of electrons passing through a cross-section of the wire in 10 s is: 
 
 
 
[1 mark] 
(a) 4.0 × 1016 
(b) 5.0 × 1017 
(c) 1.6 × 1016 
(d) 1.0 × 1017 
Answer: 
(b) 5.0 × 1017"""

stem, options, marked_answer, explanation = separate_stem_and_options(flat)
print("STEM:", stem)
print("OPTIONS:", options)
print("MARKED:", marked_answer)
print("EXP:", explanation)
