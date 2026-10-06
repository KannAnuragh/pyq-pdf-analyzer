import sys
sys.path.insert(0, "/app")
from pdf_qa_agent.ingest.option_answer_separator import classify_question_type, clean_question_options_and_answer
from pdf_qa_agent.graph.nodes import separate_stem_and_options

variations = [
    ("With parens", "What type of government is Panchayati Raj?\nA) Central government\nB) Local self-government\nC) State government\nD) Federal government\nAnswer: B"),
    ("With dots", "What type of government is Panchayati Raj?\nA. Central government\nB. Local self-government\nC. State government\nD. Federal government\nAnswer: B"),
    ("With lowercase", "What type of government is Panchayati Raj?\na) Central government\nb) Local self-government\nc) State government\nd) Federal government\nAnswer: B"),
    ("No newlines", "What type of government is Panchayati Raj? A) Central government B) Local self-government C) State government D) Federal government Answer: B"),
]

for name, content in variations:
    print(f"\n--- {name} ---")
    stem, options, marked_answer, explanation = separate_stem_and_options(content)
    print("OPTIONS LEN:", len(options))
    q_type, sub_parts = classify_question_type(options, explanation, marked_answer=marked_answer, stem=stem)
    print("CLASSIFY TYPE:", q_type)
    print("SUB PARTS:", sub_parts)
    
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
