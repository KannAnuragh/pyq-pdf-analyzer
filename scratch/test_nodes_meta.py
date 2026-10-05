import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from graph.nodes import parse_trailing_meta

clean_opt, ans, exp = parse_trailing_meta("1888\nAnswer: B")
print(f"clean_opt: {clean_opt}, ans: {ans}, exp: {exp}")
