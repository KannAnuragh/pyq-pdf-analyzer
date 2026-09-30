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

TASKS & RULES:
1. LANGUAGE & TYPOS: Fix all spelling, OCR typos, and Malayalam ligature/font errors in the stem, options, and explanation without translating.
2. GARBLED TEXT: Check the Question Stem, Options, and Explanation. If the text has minor OCR typos, correct them and return the cleaned text in `corrected_stem`, `corrected_options`, and `corrected_explanation`. BUT if ANY part of the text contains severely broken math equations (e.g. random floating numbers, "1 11", missing fraction bars), nonsensical characters, or is corrupted by bad fonts to the point you cannot confidently fix it, set `needs_gemini_ocr` to true.
3. QUESTION TYPE: Verify if question is MCQ with choices (A/B/C/D) or descriptive (single or with subquestions).
4. LATEX EQUATIONS: Format all math, fractions, and units with valid inline LaTeX ($...$). Fix broken braces ($V_{proton}$), fractions ($\\frac{a}{b}$), and units ($5 \\times 10^5\\text{ V/m}$, $20\\,\\Omega$). Never put ordinary prose or sentences inside $...$.
5. READABILITY & PARAGRAPHS: Format explanations with clean paragraph breaks (\\n\\n) and numbered steps (1., 2.). Un-mash joined words ("where $h$ is", not "whereh").
6. VERIFICATION: Independently solve and return `validated_answer` ('A'|'B'|'C'|'D' or null for descriptive), reasoning, and confidence.

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
  "reasoning": "Step-by-step reasoning",
  "corrected_stem": "Cleaned stem or original if no fixes needed",
  "corrected_options": {{"A": "Cleaned option A", ...}},
  "corrected_explanation": "Cleaned, readable explanation with proper paragraphs and LaTeX, or empty string if none provided",
  "validated_answer": "A" | "B" | "C" | "D" | null,
  "confidence": "high" | "medium" | "low",
  "needs_gemini_ocr": true | false
}}
"""
