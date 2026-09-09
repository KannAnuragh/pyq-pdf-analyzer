EXTRACT_PROMPT = """
You are a document extraction assistant. Given the following text from a PDF, extract all questions into a structured format.

Extraction Guidelines:
1. For each question, extract:
   - "question_number": int
   - "stem": The full question prompt/stem text. Convert and preserve all mathematical equations, fractions, formulas, and symbols as standard LaTeX math ($...). For matching questions (List I / List II), preserve the columns on the same line with tab (\t) or clear alignment. NEVER include the multiple-choice options (A:-, B:-, C:-, D:-) inside the stem.
   - "options": An object mapping each option letter to its text (e.g. {"A": "...", "B": "...", "C": "...", "D": "..."}). Preserve all mathematical formulas in LaTeX math ($...). Within each option, keep matching codes like '(a) (b) (c) (d) (e)' on a single line; do not split them into individual lines.
   - "marked_answer": The marked or given correct answer letter (e.g. "A", "B", "C", "D") or Malayalam answer (ഉത്തരം) if present, otherwise null.
   - "explanation": The given solution, explanation, or description printed below the question or options (e.g. following 'Solution:', 'Explanation:', 'Description:', 'വിശദീകരണം:', 'പരിഹാരം:') if present in the document, otherwise null.
2. If mathematical equations or formulas were embedded as images, transcribe them into standard inline LaTeX ($...) or display LaTeX ($$...$$), just like non-image math formulas.

Return the result as a valid JSON array matching this schema:
[
  {
    "question_number": 1,
    "stem": "Question stem text here\n(i) First statement\n(ii) Second statement",
    "options": {
      "A": "Option A text",
      "B": "Option B text",
      "C": "Option C text",
      "D": "Option D text"
    },
    "marked_answer": "B",
    "explanation": "Solution or explanation text here"
  }
]

Text:
{text}
"""

SOLVE_PROMPT = """
You are an expert solver. Given the following question stem and options, solve the question independently.
Provide your reasoning, the final answer key (e.g. A, B, C, D), and your confidence level ('high', 'medium', 'low').

Question Stem:
{stem}

Options:
{options}

Output your response as JSON matching this schema:
{
  "reasoning": "Step-by-step reasoning explaining why the answer is correct",
  "validated_answer": "A",
  "confidence": "high"
}
"""
