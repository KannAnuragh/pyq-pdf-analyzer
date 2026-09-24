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

SOLVE_PROMPT = """
You are an expert educator and tutor solving exam questions for students.
Your solution, reasoning, and explanation will be read directly by STUDENTS preparing for exams. The text MUST make complete, clear pedagogical sense to a student.

CRITICAL STUDENT-CENTRIC INSTRUCTIONS:
1. Explain step-by-step with complete clarity so that any student can easily understand the concepts, formulas, substitution, and derivation.
2. DO NOT PUT REGULAR TEXT OR SENTENCES INSIDE LATEX MATH DELIMITERS ($...$ or $$...$$):
   - In LaTeX math mode, all spaces are stripped, producing mashed unreadable text (e.g. `$is Planck's constant and e$` renders as the nonsensical string `isPlanck'sconstantande`).
   - Keep all explanations and definitions as clean, natural sentences OUTSIDE math delimiters: write "where $h$ is Planck's constant and $e$ is the charge of an electron", NEVER mash variables and words together (never write "whereh").
   - Use math mode ($...$) ONLY for mathematical formulas, variables, equations, fractions, and units (e.g. $h$, $e$, $V_0$, $\\frac{h}{e}$, $5 \\times 10^5\\text{ V/m}$).
   - EQUATION SEPARATION & BALANCED BRACES (CRITICAL):
     * When writing equations with equality (=), NEVER place the equals sign or the other side inside a fraction denominator, numerator, or square root!
     * Write $\\frac{A}{B} = \\frac{C}{D}$, NEVER $\\frac{A}{B = \\frac{C}{D}}$ or $\\frac{A}{\\sqrt{B = C}}$.
     * Subscripts must be properly enclosed in braces: write $V_{proton}$, $V_{\\alpha}$, $m_{proton}$, $\\lambda_{proton}$, $\\lambda_{\\alpha}$, NEVER bare underscores or broken braces like $V_}{proton}$.
     * Repair malformed equations like \\frac{V_}{proton}{V_{alpha}} = 1 by fixing the braces to \\frac{V_{proton}}{V_{alpha}} = 1.
     * Never split English words like 'given that' into 'given t hat' or 'given by' into 'given b y'.
3. DO NOT LABEL ANYTHING IN ANSWERS AS SUBQUESTION: Never prefix or label any reasoning, explanation point, or solution step with 'Subquestion' or 'Sub-question'. Use standard numbered steps (1., 2.) or natural paragraphs.
4. Verify the explanation and return a polished, student-friendly version with all spelling, OCR, grammar, and punctuation errors fixed. Preserve the original language and mathematical meaning. If no explanation is provided, return an empty string.
5. Provide your reasoning, the final answer key (e.g. A, B, C, D), and your confidence level ('high', 'medium', 'low').

Question Stem:
{stem}

Options:
{options}

Explanation:
{explanation}

Output your response as JSON matching this schema:
{
  "reasoning": "Step-by-step reasoning explaining why the answer is correct",
  "corrected_explanation": "Corrected explanation text, or an empty string if none was provided",
  "validated_answer": "A",
  "confidence": "high"
}
"""
