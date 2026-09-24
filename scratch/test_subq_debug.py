import sys, os
sys.path.insert(0, os.path.abspath(".."))

from pdf_qa_agent.ingest.option_answer_separator import classify_question_type, clean_question_options_and_answer

# Example 1: English subquestions with (a), (b)
sub_q_en = {
    "a": "What is photoelectric effect?",
    "b": "Write two characteristics of photoelectric effect which cannot be explained by wave theory."
}
ans_en = "(a) Emission of electrons...\n(b) (i) Instantaneous emission, (ii) Threshold frequency"

# Example 2: Malayalam subquestions
sub_q_mal = {
    "a": "പ്രകാശവൈദ്യുത പ്രഭാവം എന്നാൽ എന്ത്?",
    "b": "തരംഗസിദ്ധാന്തം വഴി വിശദീകരിക്കാൻ കഴിയാത്ത രണ്ട് സവിശേഷതകൾ എഴുതുക."
}
ans_mal = "(a) വിശദീകരണം...\n(b) വിശദീകരണം..."

# Example 3: Subquestions with question marks
sub_q_marks = {
    "a": "State Ohm's law. [1 mark]",
    "b": "Draw V-I graph for an ohmic conductor. [1 mark]"
}

# Example 4: Subquestions with A, B, C, D uppercase keys!
sub_q_abcd = {
    "A": "Define electric flux.",
    "B": "State Gauss's theorem in electrostatics.",
    "C": "Derive an expression for the electric field due to an infinitely long straight wire."
}

# Example 5: Malayalam subquestions with A, B, C keys
sub_q_mal_abc = {
    "A": "പ്രകാശവൈദ്യുത പ്രഭാവം നിർവചിക്കുക.",
    "B": "ഐൻസ്റ്റീന്റെ സമവാക്യം എഴുതുക.",
    "C": "ഇതിന്റെ രണ്ട് പ്രയോഗങ്ങൾ വിശദീകരിക്കുക."
}

# Example 6: Real MCQ
mcq_opts = {
    "A": "Coulomb",
    "B": "Newton",
    "C": "Ampere",
    "D": "Volt"
}

print("Test 1 (EN Sub):", classify_question_type(sub_q_en, ans_en))
print("Test 2 (MAL Sub):", classify_question_type(sub_q_mal, ans_mal))
print("Test 3 (Marks Sub):", classify_question_type(sub_q_marks, None))
print("Test 4 (A,B,C Sub):", classify_question_type(sub_q_abcd, None))
print("Test 5 (MAL A,B,C Sub):", classify_question_type(sub_q_mal_abc, None))
print("Test 6 (Real MCQ):", classify_question_type(mcq_opts, "Answer: B"))

# Test clean_question_options_and_answer
q_test = {
    "question_number": 5,
    "stem": "Answer the following:",
    "options": sub_q_abcd,
    "marked_answer": None,
    "explanation": "(a) Flux is ...\n(b) Gauss theorem states ...\n(c) Electric field is E = lambda / (2 pi eps r)"
}
clean_question_options_and_answer(q_test)
print("\nCleaned question result:")
print("question_type:", q_test["question_type"])
print("options:", q_test["options"])
print("sub_parts:", q_test.get("sub_parts"))
print("is_valid:", q_test.get("is_valid"))
