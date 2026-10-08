from pdf_qa_agent.ingest.option_answer_separator import clean_question_options_and_answer
import json

q = {
    "stem": "Assertion (A): ...",
    "options": {"A": "a", "B": "b"},
    "explanation": "explanation of Assertion. (c) If Assertion is true but Reason is false. (d) If both Assertion and Reason are false. Answer: (d) If both Assertion and Reason are false.",
    "question_type": "mcq"
}

q2 = clean_question_options_and_answer(q)
print(json.dumps(q2, indent=2))
