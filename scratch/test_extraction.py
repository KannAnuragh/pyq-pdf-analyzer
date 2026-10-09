import asyncio
from pdf_qa_agent.graph.nodes import separate_stem_and_options

def test_extract():
    text = """1. If the frequency of an A.C is made 4 times of its initial value, the inductive reactance will be: [1 mark]
(a) 2 times
(b) 3 times
(c) 4 times
(d) Unchanged
"""
    stem, opts, marked, exp = separate_stem_and_options(text)
    print("Stem:", stem)
    print("Options:", opts)
    print("Marked:", marked)
    print("Explanation:", exp)

if __name__ == "__main__":
    test_extract()
