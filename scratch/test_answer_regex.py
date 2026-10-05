import os
import sys
import re
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ingest.option_answer_separator import ANSWER_MARKER_PATTERN, parse_trailing_meta

text = "1888\n**Answer:** B"
print(parse_trailing_meta(text))
