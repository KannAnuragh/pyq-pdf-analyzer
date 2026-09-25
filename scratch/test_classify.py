import sys
sys.path.append('.')
from pdf_qa_agent.ingest.option_answer_separator import classify_question_type

options1 = {
    "a": "Define mutual inductance and write its SI unit.",
    "b": "Two circular loops, one of small radius r and other of larger radius R, are placed coaxially. Obtain the mutual inductance."
}

print(classify_question_type(options1))
