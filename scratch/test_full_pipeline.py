import os
import sys
import json
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock langchain modules
sys.modules['langchain_google_genai'] = mock.MagicMock()
sys.modules['langchain_core'] = mock.MagicMock()
sys.modules['langchain_core.messages'] = mock.MagicMock()

from graph.nodes import parse_questions_from_text
from ingest.option_answer_separator import clean_question_options_and_answer

content = """12. In which year was the Madras Corporation established?
A) 1588
B) 1688
C) 1788
D) 1888
Answer: B"""

extracted = parse_questions_from_text(content, 1)
print("EXTRACTED:")
print(json.dumps(extracted, indent=2))

if extracted:
    q = extracted[0]
    q = clean_question_options_and_answer(q)
    print("CLEANED:")
    print(json.dumps(q, indent=2))
