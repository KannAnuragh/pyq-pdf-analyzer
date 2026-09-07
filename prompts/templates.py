EXTRACT_PROMPT = """
You are a document extraction assistant. Given the following text from a PDF, extract all questions into a structured format.

Extraction Guidelines:
1. For each question, extract:
   - "question_number": int
   - "stem": The full question prompt/stem text. For matching questions (List I / List II), preserve the columns on the same line with tab (\t) or clear alignment. NEVER include the multiple-choice options (A:-, B:-, C:-, D:-) inside the stem.
   - "options": An object mapping each option letter to its text (e.g. {"A": "...", "B": "...", "C": "...", "D": "..."}). Within each option, keep matching codes like '(a) (b) (c) (d) (e)' on a single line; do not split them into individual lines.
   - "marked_answer": The marked or given correct answer letter (e.g. "A", "B", "C", "D") if present, otherwise null.

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
    "marked_answer": "B"
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
