import os
import sys
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph.nodes import separate_stem_and_options
from ingest.option_answer_separator import classify_question_type, clean_question_options_and_answer

content = """What type of government is Panchayati Raj?
A) Central government
B) Local self-government
C) State government
D) Federal government
Answer: B"""

stem, options, marked_answer, explanation = separate_stem_and_options(content)
print("SEPARATED OPTIONS:", options)
q_type, sub_parts = classify_question_type(options, explanation, marked_answer=marked_answer, stem=stem)
print("Q_TYPE:", q_type)

q = {
    "question_type": q_type,
    "sub_parts": sub_parts,
    "options": options,
    "stem": stem,
    "marked_answer": marked_answer,
    "explanation": explanation
}

q_clean = clean_question_options_and_answer(q)
print("CLEAN Q_TYPE:", q_clean.get("question_type"))
print("CLEAN OPTIONS:", q_clean.get("options"))
