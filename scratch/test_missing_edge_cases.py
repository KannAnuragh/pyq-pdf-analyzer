from pdf_qa_agent.graph.nodes import parse_questions_from_text, rule_check_node
from pdf_qa_agent.graph.state import GraphState

print("--- Testing Missing Question Edge Cases ---")

# Edge Case 1: Preceded by "figure."
doc_figure = """1. Find the torque as shown in the figure.
(a) 1 (b) 2 (c) 3 (d) 4

2. What is the value of electric flux?
(a) 10 (b) 20 (c) 30 (d) 40
"""
p_fig = parse_questions_from_text(doc_figure)
nums_fig = [q["question_number"] for q in p_fig]
print(f"Edge Case 1 (figure.): parsed numbers = {nums_fig}")

# Edge Case 2: Preceded by colon ":"
doc_colon = """Answer the following:
1. State Newton's first law.
(a) 1 (b) 2 (c) 3 (d) 4

2. State Newton's second law.
(a) 1 (b) 2 (c) 3 (d) 4
"""
p_colon = parse_questions_from_text(doc_colon)
nums_colon = [q["question_number"] for q in p_colon]
print(f"Edge Case 2 (preceded by colon): parsed numbers = {nums_colon}")

# Edge Case 3: Preceded by page marker <!--QAI_PAGE:1-->
doc_page = """<!--QAI_PAGE:0-->
1. What is capacitance?
(a) 1 (b) 2 (c) 3 (d) 4

<!--QAI_PAGE:1-->
2. What is inductance?
(a) 1 (b) 2 (c) 3 (d) 4
"""
p_page = parse_questions_from_text(doc_page)
nums_page = [q["question_number"] for q in p_page]
print(f"Edge Case 3 (page marker): parsed numbers = {nums_page}")

# Edge Case 4: Section Restart (Section A: 1..3, Section B: 1..3)
doc_sec = """Section A
1. First question in Section A.
(a) 1 (b) 2 (c) 3 (d) 4

2. Second question in Section A.
(a) 1 (b) 2 (c) 3 (d) 4

Section B
1. First question in Section B.
(a) 1 (b) 2 (c) 3 (d) 4

2. Second question in Section B.
(a) 1 (b) 2 (c) 3 (d) 4
"""
p_sec = parse_questions_from_text(doc_sec)
nums_sec = [q["question_number"] for q in p_sec]
print(f"Edge Case 4 (Section restart): parsed numbers = {nums_sec}, count = {len(p_sec)}")

# Edge Case 5: Rule Check deduplication with same question number from different sections
dummy_state = {
    "extracted_blocks": p_sec,
    "processed_questions": p_sec,
    "summary": {"total": 0}
}
rc_res = rule_check_node(dummy_state)
rc_blocks = rc_res.get("extracted_blocks", [])
rc_processed = rc_res.get("processed_questions", [])
print(f"Edge Case 5 (Rule Check Dedup): blocks count = {len(rc_blocks)}, processed count = {len(rc_processed)}")

# Edge Case 6: Preceded by "marks" or "time" or "date" or "by"
doc_marks = """1. Calculate energy. [5 marks]
2. Define power.
(a) J/s (b) W (c) N (d) Pa
"""
p_marks = parse_questions_from_text(doc_marks)
nums_marks = [q["question_number"] for q in p_marks]
print(f"Edge Case 6 (preceded by marks): parsed numbers = {nums_marks}")
