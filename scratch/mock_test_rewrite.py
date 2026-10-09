import json
import re

def fix_prompt():
    prompt = """
3. LATEX EQUATIONS & MATH:
   - For Descriptive: Set `corrected_options`: {}, `validated_answer`: null, `answer_status`: "none". Keep answer/solution steps ONLY in `corrected_explanation`. NEVER put solution steps into `corrected_options` or `corrected_stem`.
"""
    return prompt
