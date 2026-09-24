import sys
import json
import os

# Add to sys.path so we can import from pdf_qa_agent
sys.path.append(os.path.abspath("."))
from graph.nodes import parse_questions_from_text

with open("scratch/pdf_text.txt") as f:
    text = f.read()

questions = parse_questions_from_text(text)
print(json.dumps(questions, indent=2))
