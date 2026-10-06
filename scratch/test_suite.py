from pdf_qa_agent.graph.nodes import parse_questions_from_text, separate_stem_and_options, looks_like_question
from pdf_qa_agent.ingest.option_answer_separator import classify_question_type, is_valid_subparts, extract_subparts_from_text, clean_question_options_and_answer

print("=== Running Synthetic Test Suite ===")

# Test 1: Numeric MCQ options
q1_text = """1. What is the value of current?
(a) 10 A
(b) 20 A
(c) 30 A
(d) 40 A
Answer: (c)"""
stem1, opts1, ans1, exp1 = separate_stem_and_options(q1_text)
assert len(opts1) == 4, f"Test 1 Failed: Expected 4 options, got {opts1}"
assert ans1 == "C", f"Test 1 Failed: Expected Ans C, got {ans1}"
print("✓ Test 1: Numeric options extracted successfully (A, B, C, D).")

# Test 2: Ratio numbers must not create a false question split
doc_text = """1. An electric dipole is placed in uniform field. What is the torque?
(a) pE
(b) zero
(c) 2pE
(d) pE/2

2. Two concentric and coplanar circular loops P and Q have their radii in the ratio 2 :
3. Loop Q carries a current 9 A in the anticlockwise direction. For the magnetic field to be zero at the common centre, loop P must carry:
(a) 3 A in clockwise direction
(b) 9 A in clockwise direction
(c) 6 A in anti-clockwise direction
(d) 6 A in the clockwise direction.
Answer: (d)
"""
parsed = parse_questions_from_text(doc_text)
q_nums = [q["question_number"] for q in parsed]
assert q_nums == [1, 2], f"Test 2 Failed: Expected question numbers [1, 2], got {q_nums}"
q2 = parsed[1]
assert "ratio 2 :" in q2["stem"], "Test 2 Failed: Q2 stem did not preserve ratio"
assert len(q2["options"]) == 4, f"Test 2 Failed: Q2 should have 4 options, got {q2['options']}"
print("✓ Test 2: Ratio 2 : 3. did not create a false split; Q2 has all 4 options.")

# Test 3: Scientific units & point indicators
q3_text = """7. The circuit in figure shows two cells connected in opposition to each other. Cell E_1 is of emf 6 V and internal resistance 2 \\Omega; the cell E_2 is of emf 4 V and internal resistance 8 \\Omega. Find the potential difference between the points A and B. [2 marks]
<div class="my-3 flex flex-col items-center"><img src="https://cdn.example.com/circuit.png" alt="diagram" /></div>
Solution: Current I = (6 - 4) / (2 + 8) = 0.2 A.
V_A - V_B = 6 - 0.2 \\times 2 = 5.6 V."""
stem3, opts3, ans3, exp3 = separate_stem_and_options(q3_text)
q3_type, q3_sub = classify_question_type(opts3, exp3, ans3)
assert q3_type == "descriptive_single", f"Test 3 Failed: Expected descriptive_single, got {q3_type}"
assert q3_sub is None, f"Test 3 Failed: Expected no subparts, got {q3_sub}"
assert "circuit.png" in stem3, "Test 3 Failed: Diagram must remain in stem"
print("✓ Test 3: False split prevented for points A and B / 0.2 A.; diagram retained in stem.")

# Test 4: Genuine sub-questions with intro stem and diagram
q4_text = """8. Consider the circuit shown below:
<div class="my-3 flex flex-col items-center"><img src="https://cdn.example.com/circuit2.png" alt="diagram" /></div>
(a) State Kirchhoff's junction law.
(b) Find the current through the 5 \\Omega resistor.
Answer:
(a) Junction law states...
(b) Current is 2 A."""
q4_dict = {"question_number": 8, "stem": q4_text, "options": {}, "question_type": "descriptive"}
clean_question_options_and_answer(q4_dict)
assert q4_dict["question_type"] == "descriptive_subquestions", f"Test 4 Failed: Expected descriptive_subquestions, got {q4_dict['question_type']}"
assert "circuit2.png" in q4_dict["stem"], "Test 4 Failed: Intro diagram must remain in stem"
assert "a" in q4_dict["sub_parts"] and "b" in q4_dict["sub_parts"], f"Test 4 Failed: Expected sub_parts a and b, got {q4_dict['sub_parts']}"
print("✓ Test 4: Genuine sub-questions separated cleanly; intro diagram preserved in stem.")

# Test 5: Question starting directly with (a)
q5_text = """9. (a) Define electric flux.
(b) State Gauss's theorem and derive Coulomb's law."""
q5_dict = {"question_number": 9, "stem": q5_text, "options": {}, "question_type": "descriptive"}
clean_question_options_and_answer(q5_dict)
assert q5_dict["stem"] == "", f"Test 5 Failed: Expected empty intro stem, got {repr(q5_dict['stem'])}"
assert len(q5_dict["sub_parts"]) == 2, f"Test 5 Failed: Expected 2 subparts, got {q5_dict['sub_parts']}"
print("✓ Test 5: Question starting directly with (a) leaves stem empty without duplication.")

# Test 6: Multi-section document with gap (Section A: 1..3, Section B: 11..13)
multi_sec_doc = """
Section A
1. What is the unit of force?
(a) Newton
(b) Joule
(c) Watt
(d) Pascal
Answer: (a)

2. Define inertia.

3. State Newton's first law of motion.

Section B
11. Explain Archimedes' principle.

12. Derive the equation for kinetic energy.

13. State the law of conservation of energy.
"""
parsed_multi = parse_questions_from_text(multi_sec_doc)
q_nums_multi = [q["question_number"] for q in parsed_multi]
assert q_nums_multi == [1, 2, 3, 11, 12, 13], f"Test 6 Failed: Expected [1, 2, 3, 11, 12, 13], got {q_nums_multi}"
print("✓ Test 6: Multi-section document with Section A (1..3) and Section B (11..13) extracted all 6 questions without dropping.")

# Test 7: Parenthesized question numbers (1), (2)
paren_doc = """
(1) Explain the working of an AC generator.
(2) State Ampere's circuital law and write its mathematical formula.
"""
parsed_paren = parse_questions_from_text(paren_doc)
q_nums_paren = [q["question_number"] for q in parsed_paren]
assert q_nums_paren == [1, 2], f"Test 7 Failed: Expected [1, 2], got {q_nums_paren}"
print("✓ Test 7: Parenthesized questions (1), (2) extracted successfully.")

# Test 8: Mixed prefix markers in same paper (Question 1:, 2., (3))
mixed_doc = """
Question 1: What is electric potential?
(a) Scalar
(b) Vector
(c) Tensor
(d) Dimensionless
Answer: (a)

2. State Ohm's law.

(3) Define resistance and resistivity.
"""
parsed_mixed = parse_questions_from_text(mixed_doc)
q_nums_mixed = [q["question_number"] for q in parsed_mixed]
assert q_nums_mixed == [1, 2, 3], f"Test 8 Failed: Expected [1, 2, 3], got {q_nums_mixed}"
print("✓ Test 8: Mixed markers (Question 1:, 2., (3)) extracted all questions seamlessly.")

# Test 9: Pure math equation with MCQ choices
math_stem = "$\\int_0^1 x^2 dx = $"
math_opts = {"A": "1/3", "B": "1/2", "C": "1", "D": "0"}
assert looks_like_question(math_stem, options=math_opts, marked_answer="A"), "Test 9 Failed: Math question must not be rejected by equation filter"
print("✓ Test 9: Math equation question with MCQ choices passes looks_like_question validation.")

# Test 10: MCQ with diagrams in option choices
diag_opts = {
    "A": '<div class="my-3"><img src="https://cdn.example.com/graph_a.png" alt="diagram" /></div>',
    "B": '<div class="my-3"><img src="https://cdn.example.com/graph_b.png" alt="diagram" /></div>',
    "C": '<div class="my-3"><img src="https://cdn.example.com/graph_c.png" alt="diagram" /></div>',
    "D": '<div class="my-3"><img src="https://cdn.example.com/graph_d.png" alt="diagram" /></div>',
}
diag_q_type, diag_sub = classify_question_type(diag_opts, None, "B")
assert diag_q_type == "mcq", f"Test 10 Failed: Expected mcq, got {diag_q_type}"
assert diag_sub is None, f"Test 10 Failed: Expected no subparts for diagram MCQ, got {diag_sub}"
print("✓ Test 10: MCQ with diagram choices is strictly classified as 'mcq' and not sub-questions.")

# Test 11: Assertion-Reason question extraction (preventing Assertion (A) from becoming Option A)
from pdf_qa_agent.ingest.option_answer_separator import heal_assertion_reason_question
ar_text = """15. Assertion (A): The energy of a charged particle moving in a magnetic field does not change.
Reason (R): It is because the work done by the magnetic force on the charge moving in a magnetic field is zero. [1 mark]
(a) If both Assertion and Reason are true and Reason is correct explanation of Assertion.
(b) If both Assertion and Reason are true but Reason is not the correct explanation of Assertion.
(c) A is true, but R is false.
(d) A is false, but R is true.
Answer: (a)"""

parsed_ar = parse_questions_from_text(ar_text)
assert len(parsed_ar) == 1, f"Test 11 Failed: Expected 1 question, got {len(parsed_ar)}"
q15 = parsed_ar[0]
assert "Assertion" in q15["stem"] and "Reason" in q15["stem"], f"Test 11 Failed: Stem must contain both Assertion and Reason, got {repr(q15['stem'])}"
assert len(q15["options"]) == 4, f"Test 11 Failed: Expected 4 options, got {q15['options']}"
assert "Reason (R):" not in q15["options"]["A"], f"Test 11 Failed: Option A must not contain Reason (R):, got {repr(q15['options']['A'])}"
assert "The energy of a charged particle" not in q15["options"]["A"], f"Test 11 Failed: Option A must not contain Assertion statement, got {repr(q15['options']['A'])}"
assert "If both Assertion and Reason" in q15["options"]["A"], f"Test 11 Failed: Option A choice text corrupted, got {repr(q15['options']['A'])}"
print("✓ Test 11: Assertion (A) and Reason (R) retained in question stem; options (a)-(d) cleanly separated.")

# Test 12: Auto-healing on previously broken/trapped Assertion-Reason structure
broken_stem = "Assertion"
broken_opts = {
    "A": "The energy of a charged particle moving in a magnetic field does not change.\nReason (R): It is because the work done by the magnetic force on the charge moving in a magnetic field is zero. [1 mark]\n(a) If both Assertion and Reason are true and Reason is correct explanation of Assertion.",
    "B": "If both Assertion and Reason are true but Reason is not the correct explanation of Assertion.",
    "C": "A is true, but R is false.",
    "D": "A is false, but R is true."
}
healed_stem, healed_opts = heal_assertion_reason_question(broken_stem, broken_opts)
assert "The energy of a charged particle" in healed_stem, f"Test 12 Failed: Healed stem missing Assertion text: {healed_stem}"
assert "Reason (R):" in healed_stem, f"Test 12 Failed: Healed stem missing Reason text: {healed_stem}"
assert "The energy of a charged particle" not in healed_opts["A"], f"Test 12 Failed: Option A still contains trapped statement: {healed_opts['A']}"
# Test 13: Split merged options dictionary
from pdf_qa_agent.ingest.option_answer_separator import split_merged_options
merged_test_opts = {
    "A": "10 m/s (B) 20 m/s",
    "C": "30 m/s",
    "D": "40 m/s"
}
split_res = split_merged_options(merged_test_opts)
assert len(split_res) == 4, f"Test 13 Failed: Expected 4 options, got {split_res}"
assert split_res["A"] == "10 m/s", f"Test 13 Failed: Option A corrupted: {split_res['A']}"
assert split_res["B"] == "20 m/s", f"Test 13 Failed: Option B corrupted: {split_res['B']}"
print("✓ Test 13: split_merged_options successfully split merged (B) out of Option A.")

# Test 13b: Multi-level split where 4 options are in a single string
single_str_opts = {"A": "First (B) Second (C) Third (D) Fourth"}
split_all = split_merged_options(single_str_opts)
assert len(split_all) == 4, f"Test 13b Failed: Expected 4 options, got {split_all}"
assert split_all["A"] == "First" and split_all["D"] == "Fourth"
print("✓ Test 13b: Multi-level split successfully produced all 4 options from single string.")

# Test 14: Options ending with prepositions (e.g. 'to', 'of', 'or') not absorbing subsequent option
prep_opt_text = """1. In an AC circuit, the frequency increases
(a) from 0 to 50 Hz
(b) from 50 to 100 Hz
(c) remains constant
(d) becomes zero
Answer: (a)"""
stem_prep, opts_prep, ans_prep, exp_prep = separate_stem_and_options(prep_opt_text)
assert len(opts_prep) == 4, f"Test 14 Failed: Expected 4 options when ending in preposition, got {opts_prep}"
assert "50 to 100 Hz" in (opts_prep.get("b") or opts_prep.get("B") or ""), f"Test 14 Failed: Option B missing or corrupted: {opts_prep}"
print("✓ Test 14: Options ending in preposition 'to' do not reject (b); all 4 options extracted cleanly.")

# Test 15: Bare letter options with single space separator
bare_opts_text = """2. What is the velocity?
A. 10 m/s B. 20 m/s C. 30 m/s D. 40 m/s
Answer: B"""
stem_bare, opts_bare, ans_bare, _ = separate_stem_and_options(bare_opts_text)
assert len(opts_bare) == 4, f"Test 15 Failed: Expected 4 bare options, got {opts_bare}"
assert ans_bare == "B", f"Test 15 Failed: Expected answer B, got {ans_bare}"
print("✓ Test 15: Bare letter options (A. B. C. D.) separated cleanly without merging.")

# Test 16: Diagrams stay strictly inside their question and are not detached/hijacked
diagram_doc = """
1. Find the equivalent resistance in the circuit shown below:
<div class="my-3 flex flex-col items-center"><img src="https://cdn.example.com/resistor_circuit_1.png" alt="diagram" /></div>
(a) 10 \\Omega
(b) 20 \\Omega
(c) 30 \\Omega
(d) 40 \\Omega

2. In the following logic gate circuit, find the output Y:
<div class="my-3 flex flex-col items-center"><img src="https://cdn.example.com/logic_gate_2.png" alt="diagram" /></div>
(a) 0
(b) 1
"""
parsed_diag = parse_questions_from_text(diagram_doc)
assert len(parsed_diag) == 2, f"Test 16 Failed: Expected 2 questions, got {len(parsed_diag)}"
assert "resistor_circuit_1.png" in parsed_diag[0]["stem"], "Test 16 Failed: Diagram 1 must be in Question 1"
assert "resistor_circuit_1.png" not in parsed_diag[1]["stem"], "Test 16 Failed: Diagram 1 must NOT be in Question 2"
assert "logic_gate_2.png" in parsed_diag[1]["stem"], "Test 16 Failed: Diagram 2 must be in Question 2"
assert "logic_gate_2.png" not in parsed_diag[0]["stem"], "Test 16 Failed: Diagram 2 must NOT be in Question 1"
print("✓ Test 16: Diagrams strictly preserved inside their respective question stems without hijacking.")

# Test 17: Sequential question chain with no missing questions
chain_doc = """
1. Question one text here.
(a) 1 (b) 2 (c) 3 (d) 4

2. Question two text here.
(a) 1 (b) 2 (c) 3 (d) 4

3. Question three text here.
(a) 1 (b) 2 (c) 3 (d) 4

4. Question four text here.
(a) 1 (b) 2 (c) 3 (d) 4

5. Question five text here.
(a) 1 (b) 2 (c) 3 (d) 4
"""
parsed_chain = parse_questions_from_text(chain_doc)
chain_nums = [q["question_number"] for q in parsed_chain]
assert chain_nums == [1, 2, 3, 4, 5], f"Test 17 Failed: Expected [1, 2, 3, 4, 5], got {chain_nums}"
print("✓ Test 17: All 5 questions in sequential chain extracted with zero omissions.")

# Test 18: Question preceded by instruction ending with colon
colon_doc = """Answer the following:
1. State Newton's first law of motion.
(a) 1 (b) 2 (c) 3 (d) 4

2. State Newton's second law of motion.
(a) 1 (b) 2 (c) 3 (d) 4
"""
parsed_colon = parse_questions_from_text(colon_doc)
colon_nums = [q["question_number"] for q in parsed_colon]
assert colon_nums == [1, 2], f"Test 18 Failed: Expected [1, 2], got {colon_nums}"
print("✓ Test 18: Questions preceded by colon in instructions are never missed.")

# Test 19: Question preceded by page transition comment <!--QAI_PAGE:X-->
page_doc = """<!--QAI_PAGE:0-->
1. What is capacitance?
(a) 1 (b) 2 (c) 3 (d) 4

<!--QAI_PAGE:1-->
2. What is inductance?
(a) 1 (b) 2 (c) 3 (d) 4
"""
parsed_page = parse_questions_from_text(page_doc)
page_nums = [q["question_number"] for q in parsed_page]
assert page_nums == [1, 2], f"Test 19 Failed: Expected [1, 2], got {page_nums}"
assert parsed_page[0]["page_number"] == 0, f"Test 19 Failed: Expected page 0, got {parsed_page[0]['page_number']}"
assert parsed_page[1]["page_number"] == 1, f"Test 19 Failed: Expected page 1, got {parsed_page[1]['page_number']}"
print("✓ Test 19: Page transition markers retain questions and assign page numbers accurately.")

# Test 20: Multi-section document with repeating question numbers
sec_doc = """Section A
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
parsed_sec = parse_questions_from_text(sec_doc)
assert len(parsed_sec) == 4, f"Test 20 Failed: Expected 4 questions across sections, got {len(parsed_sec)}"
sec_nums = [q["question_number"] for q in parsed_sec]
assert sec_nums == [1, 2, 3, 4], f"Test 20 Failed: Expected continuous sequential IDs [1, 2, 3, 4], got {sec_nums}"
assert parsed_sec[2]["original_question_number"] == 1, f"Test 20 Failed: Original Q1 in Section B missing original_question_number"
print("✓ Test 20: Multi-section restarts cleanly preserved with continuous sequential IDs and original numbers.")

# Test 21: Non-destructive rule_check_node preserving all sections
from pdf_qa_agent.graph.nodes import rule_check_node
dummy_state = {
    "extracted_blocks": parsed_sec,
    "processed_questions": parsed_sec,
    "summary": {"total": 0}
}
rc_out = rule_check_node(dummy_state)
rc_blocks = rc_out["extracted_blocks"]
rc_proc = rc_out["processed_questions"]
assert len(rc_blocks) == 4, f"Test 21 Failed: Rule check dropped blocks, got {len(rc_blocks)}"
assert len(rc_proc) == 4, f"Test 21 Failed: Rule check dropped processed, got {len(rc_proc)}"
print("✓ Test 21: rule_check_node safely preserves all questions without section overwrites.")

# Test 22: Full detail preservation in schemas
from pdf_qa_agent.models import schemas
detail_doc = """1. In the circuit below, answer the following sub-parts:
<div class="my-3"><img src="https://cdn.example.com/test.png" alt="circuit" /></div>
(a) Calculate total resistance.
(b) Calculate total power.
Answer:
(a) R = 10 \\Omega
(b) P = 50 W
Explanation: By Ohm's law V = IR."""
parsed_detail = parse_questions_from_text(detail_doc)
assert len(parsed_detail) == 1, f"Test 22 Failed: Expected 1 question, got {len(parsed_detail)}"
q_det = parsed_detail[0]
qr = schemas.QuestionResult(
    question_number=q_det["question_number"],
    question_type=q_det["question_type"],
    stem=q_det["stem"],
    options=q_det["options"],
    sub_parts=q_det.get("sub_parts"),
    explanation=q_det.get("explanation"),
    answer_text=q_det.get("answer_text"),
    page_number=q_det.get("page_number")
)
assert "test.png" in qr.stem, "Test 22 Failed: Diagram missing from QuestionResult"
assert qr.sub_parts and len(qr.sub_parts) == 2, f"Test 22 Failed: Subparts missing from QuestionResult: {qr.sub_parts}"
assert qr.explanation is not None, "Test 22 Failed: Explanation missing from QuestionResult"
# Test 23: Equation multiplication and unit fraction healing
from pdf_qa_agent.ingest.math_normalizer import normalize_math_in_text, heal_equation_multiplication

t23_case1 = r"Answer:\n(b) 5 × \frac{10^5 V}{m}"
t23_out1 = normalize_math_in_text(t23_case1)
assert r"$5 \times \frac{10^{5}\text{ V}}{\text{m}}$" in t23_out1, f"Test 23 Failed on Case 1: got {t23_out1}"

t23_case2 = r"(b) 5 × $\frac{10^{5}V}{m}$"
t23_out2 = normalize_math_in_text(t23_case2)
assert r"$5 \times \frac{10^{5}\text{ V}}{\text{m}}$" in t23_out2, f"Test 23 Failed on Case 2: got {t23_out2}"

t23_case3 = r"$5 \times 10^5$V/m"
t23_out3 = normalize_math_in_text(t23_case3)
assert r"\text{ V/m}$" in t23_out3, f"Test 23 Failed on Case 3: got {t23_out3}"

t23_case4 = r"R_T = 20\Omega, V_T = 20V, I = 1A"
t23_out4 = normalize_math_in_text(t23_case4)
assert r"$R_T = 20\Omega, V_T = 20V, I = 1A$" in t23_out4, f"Test 23 Failed on Case 4: got {t23_out4}"
print("✓ Test 23: Equation multiplication, fractions, and units healed into valid KaTeX inline LaTeX.")

# Test 24: Answers and explanation steps must NEVER be labeled as subquestions
q24_opts = {
    "A": "Higher",
    "B": "Lower",
    "C": r"R_T = 20\Omega, V_T = 20V, I = 1A",
    "D": "None of the above"
}
q24_exp = """Here diode is in R.B. as current enter diode from N-side.
1. Higher
2. Lower
3. R_T = 20\\Omega, V_T = 20V, I = 1A"""
q24_ans = "C"

q24_type, q24_sub = classify_question_type(q24_opts, q24_exp, q24_ans)
assert q24_type == "descriptive_single", f"Test 24 Failed: Answer steps must be descriptive_single, got {q24_type}"
assert q24_sub is None, f"Test 24 Failed: Answer steps must NEVER be labeled as subparts: {q24_sub}"

# Verify QuestionCorrection schema supports corrected_stem and corrected_options
from pdf_qa_agent.graph.correction_node import QuestionCorrection
qc = QuestionCorrection(
    question_number=1,
    question_type="descriptive_single",
    corrected_stem="Determine the state of the circuit:",
    corrected_options={},
    validated_answer=None,
    answer_status="none"
)
assert qc.question_type == "descriptive_single"
assert qc.corrected_options == {}
assert qc.validated_answer is None
print("✓ Test 24: Answers and solution steps are never labeled as subquestions; question type is descriptive_single.")

# Test 25: Comprehensive verification of ZERO answer labeling as subquestion
from pdf_qa_agent.ingest.option_answer_separator import strip_subquestion_labels_from_answer

# 1. Answer steps in candidate parts must fail is_valid_subparts
assert not is_valid_subparts({"1": "Higher", "2": "Lower", "3": r"R_T = 20\Omega"}), "Test 25 Failed: Answer steps must be rejected by is_valid_subparts"
assert not is_valid_subparts({"a": "Higher", "b": "Lower"}), "Test 25 Failed: Answer comparative words must be rejected"
assert not is_valid_subparts({"1": "Increases", "2": "Decreases"}), "Test 25 Failed: Answer words must be rejected"
assert not is_valid_subparts({"1": "Answer: 10 V", "2": "Ans: 20 V"}), "Test 25 Failed: Answer prefixes must be rejected"

# 2. Genuine directives must pass is_valid_subparts
assert is_valid_subparts({"a": "State Ohm's law.", "b": "Calculate the resistance."}), "Test 25 Failed: Directives must pass"
assert is_valid_subparts({"a": "എന്താണ് കപ്പാസിറ്റൻസ്?", "b": "യൂണിറ്റ് എഴുതുക."}), "Test 25 Failed: Malayalam directives must pass"

# 3. extract_subparts_from_text must never extract from answers
text_with_answers = """Calculate the total current.
Answer:
(a) Higher
(b) Lower"""
assert extract_subparts_from_text(text_with_answers) is None, "Test 25 Failed: extract_subparts_from_text must not extract answer steps"

# 4. strip_subquestion_labels_from_answer cleans accidental subquestion labels while preserving content
dirty_answer = "Subquestion 1: Higher\nSubquestion 2: Lower\nSubquestion (a): 10 V\nSubquestion: Field is zero"
cleaned_ans = strip_subquestion_labels_from_answer(dirty_answer)
assert "Subquestion" not in cleaned_ans, f"Test 25 Failed: Subquestion label not stripped: {cleaned_ans}"
assert "1. Higher" in cleaned_ans and "2. Lower" in cleaned_ans and "(a) 10 V" in cleaned_ans, f"Test 25 Failed: Content mangled: {cleaned_ans}"
assert "Field is zero" in cleaned_ans, f"Test 25 Failed: Content mangled: {cleaned_ans}"

# 5. clean_question_options_and_answer with answer steps in explanation never creates sub_parts
q25_dict = {
    "question_number": 25,
    "stem": "Determine the values of the components.",
    "options": {"1": "Higher", "2": "Lower"},
    "explanation": "1. Higher\n2. Lower\n3. R_T = 20\\Omega"
}
clean_question_options_and_answer(q25_dict)
assert q25_dict["question_type"] == "descriptive_single", f"Test 25 Failed: Expected descriptive_single, got {q25_dict['question_type']}"
assert q25_dict.get("sub_parts") is None, f"Test 25 Failed: sub_parts must be None, got {q25_dict.get('sub_parts')}"
assert q25_dict["options"] == {}, f"Test 25 Failed: options must be empty, got {q25_dict['options']}"
assert "Higher" in q25_dict["explanation"] and "Lower" in q25_dict["explanation"], f"Test 25 Failed: Answer content must be preserved in explanation: {q25_dict['explanation']}"

# Test 26: Student-Centric Readability & Anti-Math-Wrapping Healing
from pdf_qa_agent.ingest.math_normalizer import heal_student_readability, normalize_math_in_text

# 1. Un-mashing words and unwrapping math-mangled prose
mangled_input = """The slope of the graph between the stopping potential ($V_0$) and the frequency ($\\nu$) is $\\frac{h}{e}$, whereh
$$isPlanck'sconstantande$$
is the charge of an electron."""

healed_output = heal_student_readability(mangled_input)
assert "where $h$ is Planck's constant and $e$ is the charge of an electron." in healed_output, \
    f"Test 26 Failed: Did not properly un-mash and unwrap prose: {healed_output}"
assert "$$isPlanck'sconstantande$$" not in healed_output, \
    f"Test 26 Failed: Mangled math block remained in output: {healed_output}"

# 2. Verify legitimate equations and symbols are preserved intact
assert "$\\frac{h}{e}$" in healed_output, "Test 26 Failed: Legitimate fraction was corrupted"
assert "($V_0$)" in healed_output, "Test 26 Failed: Legitimate variable ($V_0$) was corrupted"
assert "($\\nu$)" in healed_output, "Test 26 Failed: Legitimate variable ($\\nu$) was corrupted"

# 3. Verify inline math prose unwrapping
inline_mangled = "whereh $isPlanck'sconstantande$"
assert heal_student_readability(inline_mangled) == "where $h$ is Planck's constant and $e$", \
    f"Test 26 Failed: Inline math prose unwrap failed: {heal_student_readability(inline_mangled)}"

# 4. Verify variable un-mashing with capitalization and prepositions
assert heal_student_readability("Letx be displacement and letp be momentum.") == "Let $x$ be displacement and let $p$ be momentum."
assert heal_student_readability("withv velocity ande charge") == "with $v$ velocity and $e$ charge"

# 5. Verify Indic text unwrapped from math mode
assert heal_student_readability("ഇവിടെ $പ്രകാശവൈദ്യുത സമവാക്യം$ കാണുക.") == "ഇവിടെ പ്രകാശവൈദ്യുത സമവാക്യം കാണുക."

# 6. Full pipeline verification through normalize_math_in_text
full_pipeline_res = normalize_math_in_text(mangled_input)
assert "where $h$ is Planck's constant and $e$ is the charge of an electron." in full_pipeline_res, \
    f"Test 26 Failed: normalize_math_in_text failed to heal student readability: {full_pipeline_res}"

print("✓ Test 26: Student-Centric Readability: Prose unwrapped, mashed words un-glued, equations preserved.")

print("=== ALL 26 SYNTHETIC TESTS PASSED SUCCESSFULLY! ===")
