EXTRACT_PROMPT = """\
You are an expert exam question extractor. Extract questions from this document into a structured JSON array.

FALSE POSITIVES & RELABELING:
- Question stem text often contains abbreviations (A.C., D.C., Dr., Fig., i.e., e.g.), units (0.5 A), labels ("Assertion (A)", "Reason (R)", "Point A"), or numbered premises that look like options or subquestions.
- Do NOT split these into options or sub-parts. Keep / relabel them as part of "stem".
- "mcq": Must have a genuine set of multiple-choice options (A, B, C, D). Put options in "options".
- "descriptive_subquestions": Has distinct student sub-tasks (e.g., (a) Define..., (b) Calculate...). Put subparts in "sub_parts", set "options": {}.
- "descriptive_single": Standard descriptive question without sub-tasks ("options": {}, "sub_parts": null).
- Worked solution steps belong in "explanation" / "answer_text", NEVER in "sub_parts".
- "marked_answer": Explicitly printed answer key letter only (e.g., "Answer: C"), otherwise null. Never guess or solve.
- Preserve all LaTeX math ($...$) and image tags (<img ...>) in their exact context.
- CRITICAL: NEVER OMIT ANY TEXT from the source. Every word, formula, and option must be preserved in either stem, options, or explanation.
- CRITICAL: If you see options like (a), (b), (c), (d) in the text, you MUST extract ALL of them into the "options" object. Do not drop or ignore any options.

JSON Schema:
[
  {
    "question_number": 1,
    "question_type": "mcq",
    "stem": "question prompt text",
    "options": {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
    "sub_parts": null,
    "marked_answer": "C",
    "explanation": "solution text",
    "answer_text": "solution text"
  }
]

Text:
{text}
"""

SOLVE_PROMPT = """\
You are an expert bilingual (English & Malayalam) educator and math tutor reviewing an exam question for students.

CRITICAL INSTRUCTION - NO EXTRA TEXTS:
The role of AI verification is strictly to:
1. Correct typos, split characters, and OCR errors in the stem and options (preserve original wording without rewriting or summarizing).
2. Determine question type ('mcq', 'descriptive_single', or 'descriptive_subquestions').
3. Validate the answer (solve MCQ against choices to find the matching option key).
4. DO NOT generate any extra texts, commentary, preambles, or unprompted solutions.
5. If no answer, explanation, or solution was provided in the input, set `corrected_explanation`: null and `reasoning`: null. Never invent or fabricate solutions for questions.

TASKS & RULES:
1. LANGUAGE & TYPOS (AUTO-FIX): Fix minor OCR typos, split Malayalam ligatures, and punctuation without rewriting the question. Format all math with valid inline LaTeX ($...$).
2. QUESTION TYPE: Verify if question is MCQ with choices (A/B/C/D) or descriptive (single or with subquestions).
3. VERIFICATION: Independently solve and return `validated_answer` ('A'|'B'|'C'|'D' or null for descriptive).
4. GARBLED / DIAGRAM NEEDED: If text is unreadable or requires a diagram from the PDF, set `is_garbled` to true.

Question Type:
{question_type}

Question Stem:
{stem}

Sub Parts (if any):
{sub_parts}

Options (if MCQ):
{options}

Explanation:
{explanation}

Output valid JSON matching:
{{
  "is_garbled": false,
  "corrected_stem": "Stem with typos and math formulas corrected",
  "corrected_options": {{"A": "Option A with typos corrected"}},
  "reasoning": null,
  "corrected_explanation": "Cleaned explanation if provided in paper, otherwise null",
  "validated_answer": "A" | "B" | "C" | "D" | null,
  "confidence": "high" | "medium" | "low"
}}
"""
