import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ingest.option_answer_separator import parse_trailing_meta

print(parse_trailing_meta("1888\nAnswer: B"))
