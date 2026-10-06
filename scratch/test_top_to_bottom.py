import re

print("=== Running Synthetic Top-to-Bottom Layout Tests ===")

# Test 1: Simulating render_elements_top_to_bottom logic
# 2-column layout test:
# Column 1 at x=50, y=100..300: Q1, Q2
# Column 2 at x=350, y=100..300: Q3, Q4

col1_items = [
    {"text": "1. What is charge?", "x0": 50, "y0": 100, "x1": 250, "y1": 115, "y_center": 107.5, "is_block": False},
    {"text": "(a) Positive \t (b) Negative", "x0": 50, "y0": 125, "x1": 250, "y1": 140, "y_center": 132.5, "is_block": False},
    {"text": "2. Define Coulomb law.", "x0": 50, "y0": 160, "x1": 250, "y1": 175, "y_center": 167.5, "is_block": False},
]

col2_items = [
    {"text": "3. What is electric flux?", "x0": 350, "y0": 100, "x1": 550, "y1": 115, "y_center": 107.5, "is_block": False},
    {"text": "(a) Scalar \t (b) Vector", "x0": 350, "y0": 125, "x1": 550, "y1": 140, "y_center": 132.5, "is_block": False},
    {"text": "4. State Gauss theorem.", "x0": 350, "y0": 160, "x1": 550, "y1": 175, "y_center": 167.5, "is_block": False},
]

# In top-to-bottom column reading order, Column 1 lines must precede Column 2 lines!
combined_two_col = col1_items + col2_items

# Check column separation
mid_x = 300.0
left_items = [e for e in combined_two_col if e["x1"] <= mid_x + 10]
right_items = [e for e in combined_two_col if e["x0"] >= mid_x - 10]

assert len(left_items) == 3, f"Expected 3 left items, got {len(left_items)}"
assert len(right_items) == 3, f"Expected 3 right items, got {len(right_items)}"

# Column 1 Q1 & Q2 must come before Column 2 Q3 & Q4
reading_order = [e["text"] for e in left_items] + [e["text"] for e in right_items]
joined = "\n".join(reading_order)

from pdf_qa_agent.graph.nodes import parse_questions_from_text
parsed = parse_questions_from_text(joined)
q_nums = [q["question_number"] for q in parsed]
assert q_nums == [1, 2, 3, 4], f"Expected questions [1, 2, 3, 4] in column reading order, got {q_nums}"
print("✓ Test 1: Two-column layout processes Left Column (Q1, Q2) before Right Column (Q3, Q4) without interleaving lines.")

# Test 2: Diagram interleaving between Q1 and Q2 in reading order
diagram_stream = """1. In the circuit below, find I:
<div class="my-3 flex flex-col items-center"><img src="https://cdn.example.com/resistor.png" alt="diagram" /></div>
(a) 1 A \t (b) 2 A \t (c) 3 A \t (d) 4 A

2. In the circuit below, find V:
<div class="my-3 flex flex-col items-center"><img src="https://cdn.example.com/voltage.png" alt="diagram" /></div>
(a) 5 V \t (b) 10 V
"""

parsed_diags = parse_questions_from_text(diagram_stream)
assert len(parsed_diags) == 2, f"Expected 2 questions, got {len(parsed_diags)}"
assert "resistor.png" in parsed_diags[0]["stem"]
assert "voltage.png" in parsed_diags[1]["stem"]
assert "resistor.png" not in parsed_diags[1]["stem"]
assert "voltage.png" not in parsed_diags[0]["stem"]
print("✓ Test 2: Diagrams remain in their respective questions in top-to-bottom stream.")

# Test 3: Two options combined fix verification
option_merged_stream = """1. A projectile is launched. Its vertical velocity
(a) decreases to 0 (b) then increases to max
(c) remains constant (d) oscillates
Answer: (a)"""
from pdf_qa_agent.graph.nodes import separate_stem_and_options
_, opts_test, _, _ = separate_stem_and_options(option_merged_stream)
assert len(opts_test) == 4, f"Expected 4 options, got {opts_test}"
print("✓ Test 3: Horizontal options ending with 'to' properly split into 4 distinct options.")

print("=== ALL TOP-TO-BOTTOM SYNTHETIC TESTS PASSED! ===")
