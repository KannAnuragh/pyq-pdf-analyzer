import sys
import os
sys.path.append("/app/pdf_qa_agent")
from graph.nodes import parse_questions_from_text
import fitz

doc = fitz.open("/app/12CB CH-3 Current Electricity (DPP).pdf")
text = ""
for page in doc:
    text += page.get_text() + "\n"

questions = parse_questions_from_text(text)
print(f"Extracted {len(questions)} questions")
for q in questions:
    print(f"Q{q['question_number']}: {q['stem'][:50]}...")
