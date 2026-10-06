import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ingest.option_answer_separator import clean_question_options_and_answer

q = {
    "stem": '12. In which year was the Madras Corporation established?',
    "options": {'A': '1588', 'B': '1688', 'C': '1788', 'D': '1888\nAnswer: B'},
    "marked_answer": None,
    "explanation": None,
    "question_type": "mcq"
}

q = clean_question_options_and_answer(q)
print("CLEAN OPTIONS:", q.get("options"))
print("CLEAN QTYPE:", q.get("question_type"))
print("CLEAN MARKED:", q.get("marked_answer"))
