"""
Schema-constrained LLM block classifier.

Given a QuestionSegment (carrying LayoutElements with layout-hint metadata),
serialises it as a structured prompt and calls the configured LLM with a
strict JSON output schema to classify:

  {stem, sub_parts, options, marked_answer, explanation}

Why this beats regex:
  - "A.C. current" is not ambiguous to a model reading the sentence.
  - Malayalam questions get the same classifier — the LLM reads the language.
  - Layout hints ([BOLD], [indent=...]) are passed explicitly so the model
    can see that options are indented further than the stem.

Fallback:
  If the LLM call fails, or returns malformed JSON, the function falls back
  to separate_stem_and_options(segment.flat_text) — the existing regex path —
  so no question is ever silently lost.
"""

from __future__ import annotations

import json
import re
import traceback
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .layout_segmenter import QuestionSegment


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

BLOCK_CLASSIFY_PROMPT = """\
Parse this exam question block (English or Malayalam) into JSON.
Layout signals: [BOLD]=bold, [font=N]=font size, [indent=X]=left margin (0=left, higher=indented).

FALSE POSITIVE AWARENESS & RELABELING:
- Stem text often contains letters, abbreviations (A.C., D.C., Dr., Fig., i.e., e.g.), units (0.5 A), labels (Assertion (A), Reason (R), Point A), or numbered premises that look like options or subquestions.
- Do NOT split these into options or sub-parts. Keep / relabel them as part of "stem".
- "options" (A, B, C, D) are ONLY for true multiple-choice choices (separate lines/higher indent). If not an MCQ, set options to {}.
- "sub_parts" are ONLY for explicit sub-questions (e.g. (a) Define..., (b) Derive...). Solution steps belong in "explanation".
- "marked_answer": Explicitly printed answer key letter only (e.g. "Ans: B"), else null. Never guess.
- Preserve all LaTeX math ($...$) and tokens ({{IMG:...}}, {{EQ:...}}) verbatim in their respective fields.

Return ONLY valid JSON:
{
  "stem": "question prompt text",
  "sub_parts": {"a": "..."} | null,
  "options": {"A": "...", "B": "..."} | {},
  "marked_answer": "A" | null,
  "explanation": "solution text" | null
}

Block:
---
{structured_text}
---
"""


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def classify_block_with_llm(segment: "QuestionSegment", llm) -> Optional[dict]:
    """
    Classify a single QuestionSegment using a schema-constrained LLM call.

    Parameters
    ----------
    segment : QuestionSegment
        The question segment with layout elements.
    llm : LangChain chat model
        An initialised LLM (Groq or Gemini) from get_extract_llm().

    Returns
    -------
    dict or None
        A question dict with keys: question_number, stem, sub_parts, options,
        marked_answer, explanation, question_type, page_number, section.
        Returns None only if the segment is empty.
        Falls back to regex extraction on any LLM or parsing error.
    """
    if not segment or not segment.elements:
        return None

    # Try LLM classification first
    result = _try_llm_classify(segment, llm)

    if result is not None:
        return _finalise(result, segment)

    # LLM failed — fall back to regex path on flat text
    return _fallback_regex(segment)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _try_llm_classify(segment: "QuestionSegment", llm) -> Optional[dict]:
    """
    Call the LLM with the structured prompt. Returns the parsed JSON dict
    or None on any error.
    """
    try:
        from langchain_core.messages import HumanMessage

        structured = segment.structured_text()
        if not structured.strip():
            return None

        prompt = BLOCK_CLASSIFY_PROMPT.replace("{structured_text}", structured)
        response = llm.invoke([HumanMessage(content=prompt)])

        raw = _extract_text_content(response.content)
        cleaned = _clean_json(raw)
        parsed = json.loads(cleaned)

        if not isinstance(parsed, dict):
            return None

        # Validate that the schema has the expected top-level keys
        if "stem" not in parsed:
            return None

        return parsed

    except json.JSONDecodeError as e:
        print(
            f"[block_classifier] JSON parse error on Q{segment.q_num}: {e}. "
            "Falling back to regex.", flush=True
        )
        return None
    except Exception as e:
        print(
            f"[block_classifier] LLM call failed on Q{segment.q_num}: {e}. "
            "Falling back to regex.", flush=True
        )
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Regex fallback
# ---------------------------------------------------------------------------

def _fallback_regex(segment: "QuestionSegment") -> dict:
    """
    Use the existing separate_stem_and_options() + classify_question_type()
    pipeline on the segment's flat text.  This guarantees identical output
    to the current production path.
    """
    try:
        from ..graph.nodes import separate_stem_and_options, classify_question_type, looks_like_question
        from ..ingest.option_answer_separator import extract_subparts_from_text, is_valid_subparts

        flat = segment.flat_text
        stem, options, marked_answer, explanation = separate_stem_and_options(flat)
        question_type, sub_parts = classify_question_type(
            options, explanation, marked_answer=marked_answer
        )

        if not sub_parts and stem and question_type != "mcq":
            extracted_sub = extract_subparts_from_text(stem)
            if extracted_sub and is_valid_subparts(extracted_sub):
                sub_parts = extracted_sub
                question_type = "descriptive_subquestions"

        q_opts = {k.upper(): v for k, v in options.items()} if question_type == "mcq" else {}

        return {
            "stem": stem,
            "sub_parts": sub_parts,
            "options": q_opts,
            "marked_answer": marked_answer if question_type == "mcq" else None,
            "marked_correct_answer": marked_answer if question_type == "mcq" else None,
            "explanation": explanation,
            "answer_text": explanation,
            "question_type": question_type,
        }
    except Exception as e:
        print(f"[block_classifier] Regex fallback also failed on Q{segment.q_num}: {e}", flush=True)
        return {
            "stem": segment.flat_text,
            "sub_parts": None,
            "options": {},
            "marked_answer": None,
            "marked_correct_answer": None,
            "explanation": None,
            "answer_text": None,
            "question_type": "descriptive_single",
        }


# ---------------------------------------------------------------------------
# Finalise: merge LLM output with segment metadata and run LaTeX repair
# ---------------------------------------------------------------------------

def _finalise(llm_result: dict, segment: "QuestionSegment") -> dict:
    """
    Merge LLM-classified fields with segment metadata, normalise option keys,
    run LaTeX brace repair, and determine question_type.
    """
    stem = (llm_result.get("stem") or "").strip()
    sub_parts = llm_result.get("sub_parts") or None
    options_raw = llm_result.get("options") or {}
    marked_answer = llm_result.get("marked_answer") or None
    explanation = (llm_result.get("explanation") or "").strip() or None

    # Normalise option keys to uppercase
    options = {k.strip().upper(): v for k, v in options_raw.items() if k and v}

    # If LLM returned only 1 option (false positive like "A.C. current"), relabel it back into stem
    if options and len(options) < 2:
        for opt_k, opt_v in options.items():
            if opt_v and opt_v not in stem:
                stem = f"{stem} ({opt_k}) {opt_v}".strip()
        options = {}

    # Determine question type
    if options and len(options) >= 2:
        question_type = "mcq"
    elif sub_parts and isinstance(sub_parts, dict) and len(sub_parts) >= 1:
        question_type = "descriptive_subquestions"
    else:
        question_type = "descriptive_single"

    # Run lightweight LaTeX repair on all text fields
    try:
        from ..ingest.option_answer_separator import validate_and_repair_latex
        stem, _ = validate_and_repair_latex(stem)
        explanation, _ = validate_and_repair_latex(explanation or "")
        explanation = explanation or None
        options = {k: validate_and_repair_latex(v)[0] for k, v in options.items()}
    except Exception:
        pass

    return {
        "stem": stem,
        "sub_parts": sub_parts,
        "options": options,
        "question_type": question_type,
        "marked_answer": marked_answer if question_type == "mcq" else None,
        "marked_correct_answer": marked_answer if question_type == "mcq" else None,
        "explanation": explanation,
        "answer_text": explanation,
    }


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _extract_text_content(content) -> str:
    """Handle AIMessage.content which can be str or list of dicts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(content)


def _clean_json(raw: str) -> str:
    """Strip markdown fences and extract the JSON object."""
    raw = raw.strip()
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0]
    raw = raw.strip()
    # If the model returned a JSON array instead of object, take first element
    if raw.startswith("["):
        try:
            arr = json.loads(raw)
            if isinstance(arr, list) and arr:
                return json.dumps(arr[0])
        except Exception:
            pass
    return raw
