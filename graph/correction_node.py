"""
correction_node.py

Batched per-page LLM correction and answer verification node (spec §5–§6).

Architecture:
- One Gemini API call PER PAGE (not per word, not per question).
- Each call corrects flagged text and verifies marked answers.
- Uses structured/schema-constrained output — no free-text JSON parsing.
- Post-processing gate applies edit-distance and confidence checks.
- Answer disagreements ALWAYS route to admin review (never auto-applied).
- Accepted text corrections feed the growing correction cache.

Concurrency: pages are independent — processed with bounded parallelism.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (all tunable via env vars)
# ---------------------------------------------------------------------------

# Provider configuration: 'groq' or 'gemini' (spec §5)
# Defaults to 'groq' if ENABLE_GROQ=true, otherwise 'gemini' if ENABLE_GEMINI=true
CORRECTION_PROVIDER_ENV = os.getenv("CORRECTION_LLM_PROVIDER", "").strip().lower()

GROQ_CORRECTION_MODEL = os.getenv(
    "GROQ_CORRECTION_MODEL",
    os.getenv("GROQ_SOLVE_MODEL", os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"))
)
GEMINI_CORRECTION_MODEL = os.getenv("GEMINI_CORRECTION_MODEL", "gemini-3.1-flash-lite")

# Backward compatibility alias
CORRECTION_MODEL = GEMINI_CORRECTION_MODEL

# Post-processing gate thresholds (spec §6)
EDIT_DISTANCE_MIN_RATIO = float(os.getenv("EDIT_DISTANCE_MIN_RATIO", "60"))
CORRECTION_CONFIDENCE_FLOOR = float(os.getenv("CORRECTION_CONFIDENCE_FLOOR", "0.6"))

# Concurrency & retries
MAX_CONCURRENT_PAGES = int(os.getenv("MAX_CONCURRENT_PAGES", "5"))
MAX_RETRIES = int(os.getenv("CORRECTION_MAX_RETRIES", "3"))


def is_word_correction_enabled() -> bool:
    """Checks whether Malayalam question word correction is enabled."""
    try:
        from ..storage.config_manager import get_setting
        return bool(get_setting("word_correction_enabled", get_setting("correction_enabled", True)))
    except Exception:
        pass
    val = os.getenv("ENABLE_WORD_CORRECTION", os.getenv("ENABLE_CORRECTION", "true")).lower()
    return val in ("true", "1", "yes")


def is_answer_validation_enabled() -> bool:
    """Checks whether AI answer validation is enabled."""
    try:
        from ..storage.config_manager import get_setting
        return bool(get_setting("validation_enabled", True))
    except Exception:
        pass
    val = os.getenv("ENABLE_ANSWER_VALIDATION", os.getenv("ENABLE_VALIDATION", "true")).lower()
    return val in ("true", "1", "yes")


def get_correction_provider() -> str:
    """
    Determine which LLM provider to use for correction & validation.
    Checks runtime config manager first, then environment variables.

    Priority:
    1. If both word correction and answer validation are disabled -> 'none'
    2. Runtime config correction_provider ('groq' or 'gemini')
    3. Explicit CORRECTION_LLM_PROVIDER env var ('groq' or 'gemini')
    4. If ENABLE_GROQ=true and GROQ_API_KEY is configured -> 'groq'
    5. If ENABLE_GEMINI=true and GOOGLE_API_KEY is configured -> 'gemini'
    6. Otherwise 'none'
    """
    word_corr_on = is_word_correction_enabled()
    ans_val_on = is_answer_validation_enabled()
    if not word_corr_on and not ans_val_on:
        return "none"

    try:
        from ..storage.config_manager import get_setting
        cfg_provider = get_setting("correction_provider")
        if cfg_provider in ("groq", "gemini", "none"):
            return cfg_provider
    except Exception:
        pass

    if CORRECTION_PROVIDER_ENV in ("groq", "gemini", "none"):
        return CORRECTION_PROVIDER_ENV
    if CORRECTION_PROVIDER_ENV in ("false", "0", "no", "off"):
        return "none"

    # Priority to Groq if enabled & key configured
    groq_enabled = os.getenv("ENABLE_GROQ", "false").lower() in ("true", "1", "yes")
    groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if groq_enabled and groq_key and not groq_key.startswith("your_"):
        return "groq"

    # Gemini fallback if enabled & key configured
    gemini_enabled = os.getenv("ENABLE_GEMINI", "false").lower() in ("true", "1", "yes")
    gemini_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if gemini_enabled and gemini_key and not gemini_key.startswith("your_"):
        return "gemini"

    return "none"


def get_on_demand_llm_provider(preferred: str | None = None) -> str:
    """
    Determines an active LLM provider for explicit user button clicks
    (e.g., 'Verify with AI', 'Rewrite with AI'), even when background
    automatic validation is disabled in the settings.
    """
    if preferred and preferred in ("groq", "gemini"):
        return preferred

    try:
        from ..storage.config_manager import get_setting
        cfg_provider = get_setting("validation_provider") or get_setting("correction_provider")
        if cfg_provider in ("groq", "gemini"):
            return cfg_provider
    except Exception:
        pass

    if CORRECTION_PROVIDER_ENV in ("groq", "gemini"):
        return CORRECTION_PROVIDER_ENV

    solve_provider = os.getenv("SOLVE_LLM_PROVIDER", "").strip().lower()
    if solve_provider in ("groq", "gemini"):
        return solve_provider

    # Priority to Groq if key configured
    groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if groq_key and not groq_key.startswith("your_"):
        return "groq"

    # Gemini fallback if key configured
    gemini_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if gemini_key and not gemini_key.startswith("your_"):
        return "gemini"

    return "none"


def calculate_token_cost(prompt_tokens: int, completion_tokens: int, provider: str = "groq") -> float:
    """
    Calculates estimated cost in USD based on model pricing:
    - Groq (Llama 3.3 70B / Qwen): ~$0.59 / 1M prompt, ~$0.79 / 1M completion
    - Gemini (Gemini 2.5 Flash / 1.5 Flash): ~$0.15 / 1M prompt, ~$0.60 / 1M completion
    """
    if provider == "groq":
        cost = (prompt_tokens * 0.59 + completion_tokens * 0.79) / 1_000_000
    elif provider == "gemini":
        cost = (prompt_tokens * 0.15 + completion_tokens * 0.60) / 1_000_000
    else:
        cost = (prompt_tokens * 0.20 + completion_tokens * 0.60) / 1_000_000
    return round(cost, 6)


# ---------------------------------------------------------------------------
# Structured response schema (spec §5)
# ---------------------------------------------------------------------------

class QuestionCorrection(BaseModel):
    """Per-question structured response from the LLM."""
    question_number: int
    question_type: str = Field(
        default="mcq",
        description="The verified structural type of the question: 'mcq' | 'descriptive_single' | 'descriptive_subquestions'"
    )
    corrected_stem: str | None = Field(
        default=None,
        description="Math-corrected and typo-fixed question stem. Null if no math/typo correction needed."
    )
    corrected_options: dict[str, str] | None = Field(
        default=None,
        description="Math-corrected options dict. Empty dict {} if descriptive question. Null if no change."
    )
    corrected_words: dict[str, str] = Field(
        default_factory=dict,
        description="Map of flagged_word -> corrected_word. Empty if no correction needed."
    )
    text_confidence: float = Field(
        default=0.8,
        ge=0.0, le=1.0,
        description="Confidence in the text corrections (0.0-1.0)"
    )
    needs_ocr_fallback: bool = Field(
        default=False,
        description="Set to true if the Malayalam text is completely garbled and unreadable, requiring a full OCR rescan."
    )
    validated_answer: str | None = Field(
        default=None,
        description="The option letter (A, B, C, D) determined by the AI to be correct. MUST be null for a question with no listed options (descriptive) — never guess a letter for those."
    )
    answer_status: str = Field(
        default="agrees",
        description="'agrees' if the marked answer is correct, 'disagrees' if not, 'none' if the question is descriptive and has no options to validate against"
    )
    corrected_explanation: str | None = Field(
        default=None,
        description="The completely corrected explanation / solution text with typos and OCR errors fixed. Null if no explanation was present."
    )


class PageCorrectionResponse(BaseModel):
    """Structured response for all questions on a page."""
    corrections: list[QuestionCorrection]


class DirectQuestionCorrection(BaseModel):
    """Per-question structured response for Direct LLM Validation mode."""
    question_number: int
    question_type: str = Field(
        default="mcq",
        description="'mcq' | 'descriptive_single' | 'descriptive_subquestions'"
    )
    corrected_stem: str = Field(
        description="The completely rewritten and corrected question text (stem)."
    )
    corrected_options: dict[str, str] = Field(
        default_factory=dict,
        description="The completely rewritten and corrected options. Keys are option letters (A, B, C, D). Empty for descriptive questions."
    )
    validated_answer: str | None = Field(
        default=None,
        description="The option letter (A, B, C, D) determined by the AI to be correct. Null for descriptive."
    )
    answer_status: str = Field(
        default="agrees",
        description="'agrees' if the marked answer is correct, 'disagrees' if not. 'none' for descriptive."
    )
    corrected_explanation: str | None = Field(
        default=None,
        description="The completely rewritten and corrected explanation text. Return null if no explanation is provided."
    )


class DirectPageCorrectionResponse(BaseModel):
    corrections: list[DirectQuestionCorrection]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

CORRECTION_SYSTEM_PROMPT = r"""You are an expert bilingual (English & Malayalam) exam educator and math editor reviewing question papers for students.

TASKS & RULES:
1. LANGUAGE & TYPOS:
   - Correct all spelling, OCR errors, and flagged words in their original language without cross-translating.
   - For Malayalam: Fix split ligatures, conjuncts, chillu glyphs, and spacing. Preserve authentic Malayalam.
   - For English: Fix spelling, restore truncated words ('oth'->'Both', 'll'->'All'), and punctuation.

2. QUESTION TYPE & VERIFICATION:
   - Identify `question_type`: "mcq" (choices A/B/C/D), "descriptive_subquestions" (directive sub-tasks in stem like (a), (b), (i), (ii)), or "descriptive_single" (open-ended).
   - For MCQ: Independently solve, set `validated_answer` ('A'|'B'|'C'|'D') and `answer_status` ('agrees'|'disagrees'). Map cleaned choices in `corrected_options`.
   - For Descriptive: Set `corrected_options`: {}, `validated_answer`: null, `answer_status`: "none". Keep answer/solution steps (e.g. 1. Higher, 2. Lower) in `corrected_explanation`, not subquestions.

3. LATEX EQUATIONS & MATH:
   - Format all math, formulas, fractions, and units using valid inline LaTeX ($...$). Examples: $x^2$, $\frac{a}{b}$, $\sqrt{x}$, $5 \times 10^5\text{ V/m}$, $20\,\Omega$.
   - Fix broken braces and subscripts ($V_{proton}$, $\lambda_{\alpha}$, never $V_}{proton}$).
   - Repair fraction equations: write $\frac{A}{B} = \frac{C}{D}$, never $\frac{A}{B = \frac{C}{D}}$.
   - Use `\times` for multiplication ($5 \times 10^5$).
   - NEVER enclose normal English/Malayalam sentences in $...$ (avoids mashed text like `isPlanck'sconstantande`). Keep prose outside math delimiters.

4. PARAGRAPH FORMATTING & READABILITY:
   - Structure dense explanations and multi-point stems with clean paragraph breaks (\n\n) or numbered steps (1., 2.).
   - Un-mash concatenated words/variables (e.g. "where $h$ is", not "whereh").
   - Immutable image markers (e.g. [[QAI_IMAGE:...]]) must remain in exact positions.
   - If no explanation was in input, return null for `corrected_explanation`. Do not invent new solutions."""

JSON_CORRECTION_SYSTEM_PROMPT = f"""{CORRECTION_SYSTEM_PROMPT}

Output valid JSON matching:
{{
  "corrections": [
    {{
      "question_number": <int>,
      "question_type": "mcq" | "descriptive_single" | "descriptive_subquestions",
      "corrected_stem": "<string or null>",
      "corrected_options": {{"A": "<string>"}} or {{}} or null,
      "corrected_words": {{"<flagged_word>": "<corrected_word>"}},
      "text_confidence": <float 0.0 to 1.0>,
      "needs_ocr_fallback": <bool>,
      "validated_answer": "A" | "B" | "C" | "D" | null,
      "answer_status": "agrees" | "disagrees" | "none",
      "corrected_explanation": "<string or null>"
    }}
  ]
}}"""

DIRECT_CORRECTION_SYSTEM_PROMPT = r"""You are an expert bilingual (English & Malayalam) exam educator and math editor reviewing question papers for students.

TASKS & RULES:
1. LANGUAGE & TYPOS:
   - Correct all spelling, OCR errors, and truncated words (e.g. 'oth'->'Both', 'll'->'All').
   - For Malayalam: Fix split ligatures, conjuncts, chillu glyphs, and spacing. Do NOT translate between English and Malayalam.

2. QUESTION TYPE & VERIFICATION:
   - Identify `question_type`: "mcq" (choices A/B/C/D), "descriptive_subquestions" (directive sub-tasks (a), (b) in stem), or "descriptive_single".
   - For MCQ: Independently solve, return `validated_answer` ('A'|'B'|'C'|'D') and `answer_status` ('agrees'|'disagrees'). Put cleaned choices in `corrected_options`.
   - For Descriptive: Set `corrected_options`: {}, `validated_answer`: null, `answer_status`: "none". Never treat answer steps (1., 2.) as subquestions.

3. LATEX EQUATIONS & MATH:
   - Enclose all math, formulas, fractions, and units in valid inline LaTeX ($...$). Examples: $x^2$, $\frac{a}{b}$, $\sqrt{x}$, $5 \times 10^5\text{ V/m}$, $20\,\Omega$.
   - Fix broken braces and subscripts ($V_{proton}$, $\lambda_{\alpha}$).
   - Fix equations with fractions: write $\frac{A}{B} = \frac{C}{D}$, never $\frac{A}{B = \frac{C}{D}}$.
   - Use `\times` for multiplication ($5 \times 10^5$).
   - NEVER enclose ordinary English/Malayalam sentences inside $...$ (avoids mashed text like `isPlanck'sconstantande`). Keep prose outside math delimiters.

4. PARAGRAPH FORMATTING & READABILITY:
   - Format dense text and multi-step explanations with clean paragraph breaks (\n\n) or numbered steps (1., 2.).
   - Un-mash concatenated words/variables (write "where $h$ is", not "whereh").
   - Immutable image markers ([[QAI_IMAGE:...]]) must remain in exact positions.
   - If no explanation was in input, return null for `corrected_explanation`.

Output valid JSON matching:
{
  "corrections": [
    {
      "question_number": <int>,
      "question_type": "mcq" | "descriptive_single" | "descriptive_subquestions",
      "corrected_stem": "<string>",
      "corrected_options": {"A": "<string>"} or {},
      "corrected_explanation": "<string or null>",
      "validated_answer": "A" | "B" | "C" | "D" | null,
      "answer_status": "agrees" | "disagrees" | "none"
    }
  ]
}"""


def _clean_json_output(text: str) -> str:
    """Strips markdown code fences and extraneous text from LLM JSON response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_page_corrections(raw_text: str) -> list[QuestionCorrection]:
    """Parses JSON text into QuestionCorrection list with fallback resilience."""
    cleaned = _clean_json_output(raw_text)
    try:
        parsed = PageCorrectionResponse.model_validate_json(cleaned)
        return parsed.corrections
    except Exception:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [QuestionCorrection.model_validate(item) for item in data]
        elif isinstance(data, dict):
            for k in ("corrections", "questions", "results", "data"):
                if k in data and isinstance(data[k], list):
                    return [QuestionCorrection.model_validate(item) for item in data[k]]
        raise


FALLBACK_PLACEHOLDERS = {
    "no explanation",
    "no automated explanation available for this offline question.",
    "llm validation is disabled.",
    "llm validation skipped / offline",
    "llm call skipped or offline",
    "answer validation disabled in settings (.env)",
    "answer key verified and matches",
    "ocr fallback applied",
}

def _clean_explanation_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = str(text).strip()
    if cleaned.lower() in FALLBACK_PLACEHOLDERS:
        return ""
    # Strip leading option letter like "B\n\n..." or "(B) ..."
    cleaned = re.sub(r'^\(?([A-D])\)?[\s\.\:\-]+', '', cleaned).strip()
    # Strip leading "വിശദീകരണം:" or "Explanation:" label (only if followed by delimiter or newline)
    cleaned = re.sub(r'^(?:Solution|Explanation|Description|Sol|Exp|Hint|വിശദീകരണം|പരിഹാരം|വിവരണം|സൂചന)\s*(?:[\*\:\-]+(?:\s*|\n|$)|(?:\n|\\n))', '', cleaned, flags=re.IGNORECASE).strip()
    # Strip any accidental subquestion labels from answers
    try:
        from ..ingest.option_answer_separator import strip_subquestion_labels_from_answer
        cleaned = strip_subquestion_labels_from_answer(cleaned) or ""
    except Exception:
        pass
    # Also strip raw font glitches like ~ }
    cleaned = re.sub(r'[\~\}]+', '', cleaned)
    return cleaned


# Keep image ownership explicit while an LLM is allowed to edit surrounding text.
# A single generic placeholder was ambiguous: if a model omitted, duplicated, or
# moved it, the later positional reattachment could put a diagram in another
# question or silently append it to the wrong part of an answer.
_IMAGE_TAG_PATTERN = re.compile(
    r'(?:<div[^>]*>\s*)?<img\b[^>]*\bsrc=[\'\"][^\'\"]+[\'\"][^>]*>(?:\s*</div>)?|'
    r'\{\{IMG_BLOCK:[^}]+\}\}|\{\{IMG:[^}]+\}\}',
    re.IGNORECASE,
)
_IMAGE_TOKEN_PATTERN = re.compile(r'\[\[\s*QAI_IMAGE:[^\]]+\]\]', re.IGNORECASE)


def _mask_owned_images(text: str | None, owner: str) -> tuple[str, dict[str, str]]:
    """Replace each image with a unique, field-scoped token for an LLM prompt."""
    assets: dict[str, str] = {}

    def replace(match: re.Match) -> str:
        token = f"[[QAI_IMAGE:{owner}:{len(assets)}]]"
        assets[token] = match.group(0)
        return f"\n{token}\n"

    return _IMAGE_TAG_PATTERN.sub(replace, text or ""), assets


def _restore_owned_images(text: str | None, assets: dict[str, str], original: str | None) -> str:
    """Restore only this field's images, at their original markers.

    If the model dropped even one marker, retain the original field rather than
    guessing a new placement. This deliberately favors correct diagram
    association over a text-only LLM rewrite.
    """
    if not assets:
        return _IMAGE_TOKEN_PATTERN.sub("", text or "")

    candidate = text or ""
    if not all(token in candidate for token in assets):
        return original or ""
    # A response for this field must not contain any marker owned by a different
    # question/field. Keeping its surrounding text would still merge two answers.
    candidate_tokens = set(_IMAGE_TOKEN_PATTERN.findall(candidate))
    if not candidate_tokens.issubset(assets):
        return original or ""

    for token, image_tag in assets.items():
        candidate = candidate.replace(token, image_tag)
    # Do not allow an image token from a different question to leak into output.
    return _IMAGE_TOKEN_PATTERN.sub("", candidate)


def _build_page_prompt(
    questions_on_page: list[dict],
    flagged_words_by_question: dict[int, list[dict]],
) -> str:
    """Build a concise, token-efficient prompt for one page's worth of questions."""
    parts = []
    from ..ingest.option_answer_separator import clean_question_options_and_answer, extract_subparts_from_text

    for q in questions_on_page:
        clean_question_options_and_answer(q)
        q_num = q.get("question_number", 0)
        stem = q.get("stem", "")
        options = q.get("options", {})
        marked = q.get("marked_correct_answer") or q.get("marked_answer") or "Unknown"
        flagged = flagged_words_by_question.get(q_num, [])
        raw_exp = q.get("explanation") or q.get("answer_text") or q.get("reasoning") or ""
        explanation = _clean_explanation_text(raw_exp)

        clean_stem = q.get("_clean_stem")
        if clean_stem is None:
            clean_stem, extracted_images_stem = _mask_owned_images(stem, f"q{q_num}:stem")
            q["_extracted_images_stem"] = extracted_images_stem
            q["_clean_stem"] = clean_stem

        clean_exp = q.get("_clean_exp")
        if clean_exp is None:
            clean_exp, extracted_images_exp = _mask_owned_images(explanation, f"q{q_num}:explanation")
            q["_extracted_images_exp"] = extracted_images_exp
            q["_clean_exp"] = clean_exp

        parts.append(f"\n--- Question {q_num} ---")
        parts.append(f"Stem: {clean_stem}")
        sub_parts = q.get("sub_parts")
        if not sub_parts and clean_stem:
            sub_parts = extract_subparts_from_text(clean_stem)

        if sub_parts and isinstance(sub_parts, dict) and len(sub_parts) > 0:
            parts.append("Sub-parts:")
            for key in sorted(sub_parts.keys()):
                parts.append(f"  ({key.lower()}) {sub_parts[key]}")
        elif options:
            parts.append("Options:")
            for key in sorted(options.keys()):
                parts.append(f"  ({key}) {options[key]}")
            parts.append(f"Marked Answer: {marked}")

        if clean_exp.strip():
            parts.append(f"Explanation: {clean_exp.strip()}")

        if flagged:
            parts.append("Flagged words to check:")
            for fw in flagged:
                parts.append(f"  - '{fw['word']}' ({fw.get('reason', 'typo')})")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Edit-distance sanity check (spec §6)
# ---------------------------------------------------------------------------

def _passes_edit_distance_check(original: str, corrected: str) -> bool:
    """
    Rejects corrections where the corrected text differs from the original
    by more than ~40% of its characters. That's not a typo fix — it's a
    different piece of text.

    Uses rapidfuzz if available, falls back to a simple ratio calculation.
    """
    if original == corrected:
        return True

    try:
        from rapidfuzz import fuzz
        ratio = fuzz.ratio(original, corrected)
        return ratio >= EDIT_DISTANCE_MIN_RATIO
    except ImportError:
        # Fallback: simple character-level similarity
        if not original or not corrected:
            return False
        max_len = max(len(original), len(corrected))
        common = sum(1 for a, b in zip(original, corrected) if a == b)
        ratio = (common / max_len) * 100
        return ratio >= EDIT_DISTANCE_MIN_RATIO


# ---------------------------------------------------------------------------
# Single-page LLM calls with retries (Groq and Gemini)
# ---------------------------------------------------------------------------

async def _call_groq_for_page(
    questions_on_page: list[dict],
    flagged_words_by_question: dict[int, list[dict]],
) -> tuple[list[QuestionCorrection] | None, int, int]:
    """
    Make one batched Groq call for all flagged questions on a page.
    Uses json_object response format with schema instruction.
    Returns (corrections, prompt_tokens, completion_tokens).
    """
    try:
        from groq import AsyncGroq
    except ImportError:
        logger.error("[CorrectionNode] 'groq' package not installed — skipping Groq correction")
        return None, 0, 0

    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key or api_key.startswith("your_"):
        logger.warning("[CorrectionNode] GROQ_API_KEY not configured — skipping Groq correction")
        return None, 0, 0

    prompt = _build_page_prompt(questions_on_page, flagged_words_by_question)
    if not prompt.strip():
        return [], 0, 0

    client = AsyncGroq(api_key=api_key)
    model = GROQ_CORRECTION_MODEL

    for attempt in range(MAX_RETRIES):
        try:
            max_tok = 4096
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JSON_CORRECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=max_tok,
            )

            p_tok = response.usage.prompt_tokens if hasattr(response, "usage") and response.usage else 0
            c_tok = response.usage.completion_tokens if hasattr(response, "usage") and response.usage else 0

            raw_text = response.choices[0].message.content or ""
            if not raw_text.strip():
                logger.warning("[CorrectionNode] Empty response from Groq (attempt %d)", attempt + 1)
                continue

            corrections = _parse_page_corrections(raw_text)
            return corrections, p_tok, c_tok

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
            is_transient = is_rate_limit or "503" in err_str or "500" in err_str or "timeout" in err_str.lower()

            if is_transient and attempt < MAX_RETRIES - 1:
                wait = (attempt + 1) * (4 if is_rate_limit else 2)
                logger.warning(
                    "[CorrectionNode] Transient error from Groq (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, MAX_RETRIES, wait, err_str[:200]
                )
                await asyncio.sleep(wait)
            else:
                logger.error("[CorrectionNode] Groq LLM call failed after %d attempts: %s", attempt + 1, e)
                return None, 0, 0

    return None, 0, 0


async def _call_gemini_for_page(
    questions_on_page: list[dict],
    flagged_words_by_question: dict[int, list[dict]],
) -> tuple[list[QuestionCorrection] | None, int, int]:
    """
    Make one batched Gemini call for all flagged questions on a page.
    Uses schema-constrained output for reliable structured responses.
    Returns (corrections, prompt_tokens, completion_tokens).
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error("[CorrectionNode] 'google-genai' package not installed — skipping Gemini correction")
        return None, 0, 0

    api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key or api_key.startswith("your_"):
        logger.warning("[CorrectionNode] GOOGLE_API_KEY not configured — skipping Gemini correction")
        return None, 0, 0

    prompt = _build_page_prompt(questions_on_page, flagged_words_by_question)
    if not prompt.strip():
        return [], 0, 0

    client = genai.Client(api_key=api_key)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=GEMINI_CORRECTION_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)]
                    )
                ],
                config=types.GenerateContentConfig(
                    system_instruction=JSON_CORRECTION_SYSTEM_PROMPT,
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )

            p_tok = response.usage_metadata.prompt_token_count if hasattr(response, "usage_metadata") and response.usage_metadata else 0
            c_tok = response.usage_metadata.candidates_token_count if hasattr(response, "usage_metadata") and response.usage_metadata else 0

            raw_text = response.text
            if not raw_text:
                logger.warning("[CorrectionNode] Empty response from Gemini (attempt %d)", attempt + 1)
                continue

            corrections = _parse_page_corrections(raw_text)
            return corrections, p_tok, c_tok

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            is_transient = is_rate_limit or "503" in err_str or "500" in err_str

            if is_transient and attempt < MAX_RETRIES - 1:
                wait = (2 ** attempt) * (2 if is_rate_limit else 1)
                logger.warning(
                    "[CorrectionNode] Transient error from Gemini (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1, MAX_RETRIES, wait, err_str[:200]
                )
                await asyncio.sleep(wait)
            else:
                logger.error("[CorrectionNode] Gemini LLM call failed after %d attempts: %s", attempt + 1, e)
                return None, 0, 0

    return None, 0, 0


async def _call_llm_for_page(
    questions_on_page: list[dict],
    flagged_words_by_question: dict[int, list[dict]],
    provider: str | None = None,
) -> tuple[list[QuestionCorrection] | None, int, int]:
    """
    Dispatches the per-page correction & verification call to the active LLM provider.
    Returns (corrections, prompt_tokens, completion_tokens).
    """
    if provider is None:
        provider = get_correction_provider()

    if provider == "groq":
        logger.info("[CorrectionNode] Using Groq (%s) for page validation & correction", GROQ_CORRECTION_MODEL)
        res, p_tok, c_tok = await _call_groq_for_page(questions_on_page, flagged_words_by_question)
        if res is not None:
            return res, p_tok, c_tok
        gemini_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
        if gemini_key and not gemini_key.startswith("your_"):
            logger.warning("[CorrectionNode] Groq failed; attempting fallback to Gemini")
            return await _call_gemini_for_page(questions_on_page, flagged_words_by_question)
        return None, 0, 0
    elif provider == "gemini":
        logger.info("[CorrectionNode] Using Gemini (%s) for page validation & correction", GEMINI_CORRECTION_MODEL)
        res, p_tok, c_tok = await _call_gemini_for_page(questions_on_page, flagged_words_by_question)
        if res is not None:
            return res, p_tok, c_tok
        groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if groq_key and not groq_key.startswith("your_"):
            logger.warning("[CorrectionNode] Gemini failed or quota exhausted; attempting fallback to Groq")
            return await _call_groq_for_page(questions_on_page, flagged_words_by_question)
        return None, 0, 0
    else:
        logger.warning("[CorrectionNode] No LLM provider available for correction & validation")
        return None, 0, 0


# ---------------------------------------------------------------------------
# Post-processing gate (spec §6)
# ---------------------------------------------------------------------------

def _apply_post_processing_gate(
    corrections: list[QuestionCorrection],
    questions_by_number: dict[int, dict],
    job_id: str,
) -> tuple[list[dict], list[dict], list[int]]:
    """
    Apply the post-processing gate to LLM corrections.

    Returns:
        (accepted_corrections, review_flags, ocr_fallbacks)

    Rules:
    - Text corrections: edit-distance check + confidence floor → accept or reject
    - Answer disagreements: ALWAYS flagged for admin, never auto-applied
    """
    from ..storage.models import ReviewFlag
    from ..storage.database import SessionLocal

    accepted = []
    flags = []
    ocr_fallbacks = []

    for correction in corrections:
        q_num = correction.question_number
        q = questions_by_number.get(q_num, {})

        stem = q.get("stem", "")
        from ..ingest.quality_check import is_question_unreadable
        if (
            getattr(correction, 'needs_ocr_fallback', False)
            or q.get("is_garbled", False)
            or q.get("ocr_uncertain", False)
            or is_question_unreadable(stem, q.get("options", {}))
        ):
            ocr_fallbacks.append(q_num)

        # --- Text correction gate ---
        for original, corrected in correction.corrected_words.items():
            if original == corrected:
                continue

            passes_edit = _passes_edit_distance_check(original, corrected)
            passes_confidence = correction.text_confidence >= CORRECTION_CONFIDENCE_FLOOR

            if passes_edit and passes_confidence:
                accepted.append({
                    "question_number": q_num,
                    "original": original,
                    "corrected": corrected,
                    "confidence": correction.text_confidence,
                })
            else:
                reason_parts = []
                if not passes_edit:
                    reason_parts.append("edit_distance_exceeded")
                if not passes_confidence:
                    reason_parts.append(f"low_confidence({correction.text_confidence:.2f})")

                flags.append({
                    "job_id": job_id,
                    "question_number": q_num,
                    "flag_type": "text_rejection",
                    "original_text": original,
                    "proposed_text": corrected,
                    "llm_explanation": f"Rejected: {', '.join(reason_parts)}",
                    "confidence": correction.text_confidence,
                })

        # --- Answer verification gate (asymmetric — always flags) ---
        if correction.answer_status == "disagrees":
            flags.append({
                "job_id": job_id,
                "question_number": q_num,
                "flag_type": "answer_disagreement",
                "original_text": q.get("marked_answer", ""),
                "proposed_text": None,
                "llm_explanation": "LLM disagrees with marked answer",
                "confidence": correction.text_confidence,
            })

    # Persist review flags to DB
    if flags:
        db = SessionLocal()
        try:
            for flag_data in flags:
                db_flag = ReviewFlag(
                    job_id=flag_data["job_id"],
                    question_number=flag_data["question_number"],
                    flag_type=flag_data["flag_type"],
                    original_text=flag_data.get("original_text"),
                    proposed_text=flag_data.get("proposed_text"),
                    llm_explanation=flag_data.get("llm_explanation"),
                    confidence=flag_data.get("confidence"),
                )
                db.add(db_flag)
            db.commit()
            logger.info("[CorrectionNode] Persisted %d review flags for job %s", len(flags), job_id)
        except Exception:
            db.rollback()
            logger.exception("[CorrectionNode] Failed to persist review flags")
        finally:
            db.close()

    return accepted, flags, ocr_fallbacks


# ---------------------------------------------------------------------------
# Cache write-back (spec §4 + §9.5)
# ---------------------------------------------------------------------------

def _write_back_to_cache(
    accepted: list[dict],
    page_font_hashes: dict[str, str],
    document_id: str,
) -> int:
    """
    Write accepted corrections to the staging cache.

    Only corrections that passed the edit-distance and confidence gates
    are written here.  The cache itself handles promotion from staging
    to trusted after sufficient agreeing occurrences.

    Returns the number of entries that were promoted to trusted.
    """
    from ..ingest.correction_cache import get_correction_cache

    cache = get_correction_cache()
    promoted_count = 0

    # Use the first available font hash as the cache key
    # (in practice, most pages use one primary font)
    font_hash = next(iter(page_font_hashes.values()), "unknown")

    for item in accepted:
        promoted = cache.stage(
            font_hash=font_hash,
            garbled_span=item["original"],
            correction=item["corrected"],
            document_id=document_id,
        )
        if promoted:
            promoted_count += 1

    return promoted_count


# ---------------------------------------------------------------------------
# Batched LLM processing (keeps requests well under rate limits)
# ---------------------------------------------------------------------------

BATCH_SIZE = int(os.getenv("CORRECTION_BATCH_SIZE", "5"))


async def _call_llm_batched(
    questions_with_flags: list[dict],
    all_flagged: dict[int, list[dict]],
    provider: str,
) -> tuple[list[QuestionCorrection], int, int]:
    """
    Process flagged questions in small batches.
    Defaults to 2 questions/batch for Groq (to strictly stay under 7,000 ITPM and 1,000 OTPM limits).
    Returns (all_corrections, total_prompt_tokens, total_completion_tokens).
    """
    all_corrections: list[QuestionCorrection] = []
    total_p_tok = 0
    total_c_tok = 0

    batch_size = int(os.getenv("CORRECTION_BATCH_SIZE", "2" if provider == "groq" else "5"))
    pause_sec = 2.0 if provider == "groq" else 0.5

    total_batches = (len(questions_with_flags) + batch_size - 1) // batch_size
    for idx, i in enumerate(range(0, len(questions_with_flags), batch_size)):
        batch = questions_with_flags[i:i + batch_size]
        batch_flagged = {
            q["question_number"]: all_flagged[q["question_number"]]
            for q in batch
            if q.get("question_number") in all_flagged
        }
        print(
            f"[CorrectionNode] Processing batch {idx + 1}/{total_batches} ({len(batch)} questions)...",
            flush=True
        )
        logger.info(
            "[CorrectionNode] Processing batch %d/%d (%d questions)",
            idx + 1, total_batches, len(batch)
        )
        batch_corrections, p_tok, c_tok = await _call_llm_for_page(batch, batch_flagged, provider)
        total_p_tok += p_tok
        total_c_tok += c_tok
        if batch_corrections:
            all_corrections.extend(batch_corrections)

        # Pause between batches to respect rate limits
        if i + batch_size < len(questions_with_flags):
            await asyncio.sleep(pause_sec)

    return all_corrections, total_p_tok, total_c_tok


# ---------------------------------------------------------------------------
# Main graph node
# ---------------------------------------------------------------------------

def correction_and_validation_node(state: dict) -> dict:
    """
    LangGraph node: batched per-page correction + answer verification.

    Pipeline:
    1. Route words (cache hit → skip, dict miss / low confidence → flag)
    2. Group flagged questions by page
    3. One LLM call per page
    4. Post-processing gate (edit-distance, confidence, answer asymmetry)
    5. Cache write-back for accepted corrections
    6. Apply accepted corrections to question text

    Runs synchronously from the graph's perspective but uses asyncio
    internally for concurrent page processing.
    """
    from ..ingest.word_router import route_words_for_page
    from ..ingest.correction_cache import get_correction_cache

    extracted_blocks = state.get("extracted_blocks", [])
    pages = state.get("pages", [])
    job_id = state.get("file_path", "unknown")

    if not extracted_blocks:
        return {
            "flagged_words": {},
            "correction_results": [],
            "review_flags": [],
        }

    cache = get_correction_cache()

    force_validation = bool(state.get("force_validation", False))
    if force_validation:
        ans_val_on = True
        word_corr_on = True
        provider = state.get("provider") or get_on_demand_llm_provider()
        llm_enabled = provider != "none"
    else:
        provider = get_correction_provider()
        llm_enabled = provider != "none"
        word_corr_on = is_word_correction_enabled()
        ans_val_on = is_answer_validation_enabled()

    # ---- Step 1: Route words for all questions (if word correction enabled) ----
    all_flagged: dict[int, list[dict]] = {}
    all_cache_applied: list[dict] = []
    corrected_blocks = []

    if word_corr_on:
        for q in extracted_blocks:
            q_num = q.get("question_number", 0)

            # Find the page this question belongs to
            # (heuristic: use the first page that has matching font data)
            page_meta = pages[0] if pages else {}
            page_font_hashes = page_meta.get("font_hashes", {})
            char_confidences = page_meta.get("char_confidences", {})

            # Build full text for routing
            parts = [q.get("stem", "")]
            options = q.get("options", {})
            for key in sorted(options.keys()):
                parts.append(f"{key}. {options[key]}")
            full_text = "\n".join(parts)

            routing = route_words_for_page(
                questions=[q],
                page_font_hashes=page_font_hashes,
                char_confidences=char_confidences,
                cache=cache,
                document_id=job_id,
            )

            for r in routing:
                if r.flagged_words:
                    all_flagged[r.question_number] = [
                        {
                            "word": fw.word,
                            "position": fw.position,
                            "reason": fw.reason,
                            "context": fw.context,
                        }
                        for fw in r.flagged_words
                    ]
                all_cache_applied.extend(r.cache_applied)

                # Update question text with cache corrections
                corrected_q = dict(q)
                if r.cache_applied:
                    corrected_stem = corrected_q.get("stem", "")
                    corrected_opts = dict(corrected_q.get("options", {}))
                    corrected_exp = corrected_q.get("explanation") or corrected_q.get("answer_text") or ""
                    for applied in r.cache_applied:
                        garbled = applied["garbled"]
                        correction = applied["correction"]
                        corrected_stem = corrected_stem.replace(garbled, correction)
                        for opt_key in corrected_opts:
                            corrected_opts[opt_key] = corrected_opts[opt_key].replace(garbled, correction)
                        if corrected_exp:
                            corrected_exp = corrected_exp.replace(garbled, correction)
                    corrected_q["stem"] = corrected_stem
                    corrected_q["options"] = corrected_opts
                    if corrected_exp:
                        corrected_q["explanation"] = corrected_exp
                        corrected_q["answer_text"] = corrected_exp
                corrected_blocks.append(corrected_q)
    else:
        logger.info(
            "[CorrectionNode] Question word correction is DISABLED in .env (ENABLE_WORD_CORRECTION=false) — skipping dictionary & cache replacement"
        )
        corrected_blocks = [dict(q) for q in extracted_blocks]

    # ---- Step 2: If no LLM enabled, skip LLM ----
    if not llm_enabled:
        logger.info(
            "[CorrectionNode] No LLM provider is active — skipping LLM correction and validation"
        )
        
        # Populate processed_questions with unvalidated status and complete pipeline_steps
        processed = []
        for q in corrected_blocks:
            marked = q.get("marked_correct_answer") or q.get("marked_answer")
            has_math_img = "<img" in q.get("stem", "") or any("<img" in str(v) for v in q.get("options", {}).values())
            has_latex = "$" in q.get("stem", "") or "\\frac" in q.get("stem", "")

            cache_status = "completed" if word_corr_on else "skipped"
            cache_details = "Word correction disabled in .env (ENABLE_WORD_CORRECTION=false)" if not word_corr_on else "Passed dictionary & font cache validation"

            p_steps = {
                "scanned": {
                    "name": "scanned",
                    "label": "Scanned",
                    "status": "completed",
                    "details": f"Extracted from PDF Page {(q.get('page_number') or 0) + 1}"
                },
                "checked_with_caches": {
                    "name": "checked_with_caches",
                    "label": "Checked with Caches",
                    "status": cache_status,
                    "details": cache_details
                },
                "send_errors_to_llms": {
                    "name": "send_errors_to_llms",
                    "label": "Send Errors to LLMs",
                    "status": "skipped",
                    "details": "AI validation skipped (click 'Verify with AI' to validate)"
                },
                "ocr_scanned": {
                    "name": "ocr_scanned",
                    "label": "OCR Scanned",
                    "status": "completed" if (has_math_img or has_latex) else "not_needed",
                    "details": "Math equation clips OCR parsed to LaTeX" if has_math_img else ("Math expressions parsed" if has_latex else "Text-only question (no image OCR needed)")
                }
            }

            exp_val = q.get("explanation") or q.get("answer_text") or ""
            reasoning = exp_val if exp_val.strip() else "Unvalidated (click 'Verify with AI' to validate)"

            processed.append({
                **q,
                "marked_correct_answer": marked,
                "marked_answer": marked,
                "validated_answer": None,
                "is_valid": False,
                "confidence": "none",
                "explanation": exp_val if exp_val.strip() else None,
                "answer_text": exp_val if exp_val.strip() else None,
                "reasoning": reasoning,
                "used_web_search": False,
                "pipeline_steps": p_steps
            })
            
        summary = {
            "total": len(processed),
            "validated": 0,
            "flagged": 0,
            "unvalidated": len(processed),
            "gemini_enabled": False,
            "answer_validation_enabled": False,
            "token_usage": state.get("token_usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0
            }),
        }

        return {
            "extracted_blocks": corrected_blocks or extracted_blocks,
            "processed_questions": processed,
            "summary": summary,
            "token_usage": summary["token_usage"],
            "flagged_words": all_flagged,
            "correction_results": [],
            "review_flags": [],
        }

    # ---- Step 3: Group by page and call LLM in batches ----
    logger.info(
        "[CorrectionNode] Calling %s unconditionally for correction & validation on %d questions",
        provider.upper(), len(corrected_blocks)
    )

    # Mask large base64 images from stems and explanations to avoid leaking tens of thousands of tokens
    from ..ingest.option_answer_separator import clean_question_options_and_answer
    for q in corrected_blocks:
        clean_question_options_and_answer(q)
        stem = q.get("stem", "")
        clean_stem, extracted_images_stem = _mask_owned_images(stem, f"q{q.get('question_number', 0)}:stem")
        q["_extracted_images_stem"] = extracted_images_stem
        q["_clean_stem"] = clean_stem

        explanation = q.get("explanation") or q.get("answer_text") or ""
        clean_exp, extracted_images_exp = _mask_owned_images(explanation, f"q{q.get('question_number', 0)}:explanation")
        q["_extracted_images_exp"] = extracted_images_exp
        q["_clean_exp"] = clean_exp

    questions_by_number = {q["question_number"]: q for q in corrected_blocks}

    # Run the async LLM call unconditionally for all questions
    loop = None
    llm_p_tokens = 0
    llm_c_tokens = 0
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                corrections, llm_p_tokens, llm_c_tokens = pool.submit(
                    asyncio.run,
                    _call_llm_batched(corrected_blocks, all_flagged, provider)
                ).result()
        else:
            corrections, llm_p_tokens, llm_c_tokens = loop.run_until_complete(
                _call_llm_batched(corrected_blocks, all_flagged, provider)
            )
    except RuntimeError:
        corrections, llm_p_tokens, llm_c_tokens = asyncio.run(
            _call_llm_batched(corrected_blocks, all_flagged, provider)
        )

    if not corrections:
        # Fallback if LLM fails
        processed = []
        for q in corrected_blocks:
            marked = q.get("marked_correct_answer") or q.get("marked_answer")
            exp_val = q.get("explanation") or q.get("answer_text") or ""
            reasoning = exp_val if exp_val.strip() else "no explanation"
            processed.append({
                **q,
                "marked_correct_answer": marked,
                "marked_answer": marked,
                "validated_answer": None,
                "is_valid": False,
                "confidence": "none",
                "explanation": exp_val if exp_val.strip() else None,
                "answer_text": exp_val if exp_val.strip() else None,
                "reasoning": reasoning,
                "used_web_search": False,
            })
            
        return {
            "extracted_blocks": corrected_blocks,
            "processed_questions": processed,
            "flagged_words": all_flagged,
            "correction_results": [],
            "review_flags": [],
        }

    # ---- Step 4: Post-processing gate ----
    accepted, review_flags, ocr_fallbacks = _apply_post_processing_gate(
        corrections, questions_by_number, job_id
    )

    # ---- Step 5: Cache write-back ----
    page_font_hashes = pages[0].get("font_hashes", {}) if pages else {}
    promoted = _write_back_to_cache(accepted, page_font_hashes, job_id)
    if promoted:
        logger.info("[CorrectionNode] %d corrections promoted to trusted cache", promoted)

    # ---- Step 5.5: OCR Fallback for completely unreadable questions ----
    fallback_p_tokens = 0
    fallback_c_tokens = 0
    if ocr_fallbacks:
        from ..graph.nodes import rescan_question_vision, _solve_single_block, parse_questions_from_text
        from ..storage.config_manager import get_setting
        import pymupdf

        try:
            streak_threshold = int(get_setting("auto_whole_pdf_ocr_threshold", 3))
        except Exception:
            streak_threshold = int(os.getenv("AUTO_WHOLE_PDF_OCR_THRESHOLD", "3"))

        sorted_fallbacks = sorted(set(ocr_fallbacks))
        longest_streak = []
        cur_streak = []
        for qn in sorted_fallbacks:
            if not cur_streak or qn == cur_streak[-1] + 1:
                cur_streak.append(qn)
                if len(cur_streak) > len(longest_streak):
                    longest_streak = list(cur_streak)
            else:
                cur_streak = [qn]

        if streak_threshold > 0 and len(longest_streak) >= streak_threshold and os.path.exists(job_id):
            logger.info(
                "[CorrectionNode] Detected %d consecutive unreadable questions flagged by LLM (Q%d-Q%d). "
                "Escalating to full PDF OCR scan...",
                len(longest_streak), longest_streak[0], longest_streak[-1]
            )
            from ..ingest.router import ocr_whole_pdf
            try:
                ocr_engine = get_setting("auto_whole_pdf_ocr_engine", "auto")
            except Exception:
                ocr_engine = os.getenv("AUTO_WHOLE_PDF_OCR_ENGINE", "auto")

            ocr_res = ocr_whole_pdf(job_id, engine=ocr_engine)
            fallback_p_tokens += ocr_res.get("prompt_tokens", 0)
            fallback_c_tokens += ocr_res.get("completion_tokens", 0)

            recovered_blocks = parse_questions_from_text(ocr_res["full_text"])
            recovered_map = {b["question_number"]: b for b in recovered_blocks}

            for q in corrected_blocks:
                q_num = q.get("question_number")
                if q_num in recovered_map:
                    rec = recovered_map[q_num]
                    q["stem"] = rec.get("stem", q["stem"])
                    q["options"] = rec.get("options", q.get("options", {}))
                    if rec.get("marked_answer"):
                        q["marked_answer"] = rec["marked_answer"]
                    try:
                        solved = _solve_single_block(q, 1)
                        q["_ocr_solved_result"] = solved
                    except Exception as e:
                        logger.error("[CorrectionNode] Failed to re-solve recovered Q%d: %s", q_num, e)
        else:
            logger.info("[CorrectionNode] Executing dynamic targeted OCR fallback for %d questions", len(ocr_fallbacks))
            try:
                doc = pymupdf.open(job_id)
                
                def _get_crop(page, qn: int):
                    blocks = page.get_text("blocks")
                    start_y = None; end_y = None
                    q_start = re.compile(fr'(?:^|\n)\s*{qn}\s*[\.\):-]\s+')
                    next_q = re.compile(fr'(?:^|\n)\s*{qn + 1}\s*[\.\):-]\s+')
                    for b in blocks:
                        txt = b[4]
                        if start_y is None and q_start.search(txt):
                            start_y = max(0, b[1] - 12)
                        elif start_y is not None and next_q.search(txt):
                            end_y = b[1] - 5; break
                    if start_y is not None:
                        if end_y is None: end_y = max(start_y + 100, page.rect.height - 30)
                        return pymupdf.Rect(0, start_y, page.rect.width, end_y)
                    return None

                for q in corrected_blocks:
                    q_num = q.get("question_number")
                    if q_num in ocr_fallbacks:
                        page_idx = q.get("page_number") or 0
                        if page_idx < len(doc):
                            page = doc[page_idx]
                            crop = _get_crop(page, q_num)
                            pix = page.get_pixmap(clip=crop, dpi=200) if crop else page.get_pixmap(dpi=200)
                            image_bytes = pix.tobytes("png")
                            
                            try:
                                ext_data, p_tok, c_tok = rescan_question_vision(image_bytes, q_num, engine="gemini")
                                fallback_p_tokens += p_tok
                                fallback_c_tokens += c_tok
                                
                                # Replace content
                                q["stem"] = ext_data.get("stem", q["stem"])
                                q["options"] = ext_data.get("options", q.get("options", {}))
                                
                                # Retain marking if not extracted
                                if not ext_data.get("marked_answer"):
                                    ext_data["marked_answer"] = q.get("marked_answer")
                                    
                                # solve again with new text
                                solved = _solve_single_block(ext_data, 1)
                                
                                # Store solved result for Step 6
                                q["_ocr_solved_result"] = solved
                            except Exception as e:
                                logger.error("[CorrectionNode] OCR fallback failed for Q%d: %s", q_num, e)
                doc.close()
            except Exception as e:
                logger.error("[CorrectionNode] Failed to open PDF for OCR fallback: %s", e)


    # ---- Step 6: Apply accepted corrections and format processed_questions ----
    processed = []
    
    # First, apply text corrections if word correction is enabled
    if word_corr_on:
        for item in accepted:
            q_num = item["question_number"]
            original = item["original"]
            corrected = item["corrected"]

            for q in corrected_blocks:
                if q.get("question_number") == q_num:
                    q["stem"] = q.get("stem", "").replace(original, corrected)
                    opts = q.get("options", {})
                    for opt_key in opts:
                        opts[opt_key] = opts[opt_key].replace(original, corrected)
                    if q.get("sub_parts") and isinstance(q["sub_parts"], dict):
                        for sp_key in q["sub_parts"]:
                            q["sub_parts"][sp_key] = q["sub_parts"][sp_key].replace(original, corrected)
                    if q.get("explanation"):
                        q["explanation"] = q["explanation"].replace(original, corrected)
                    if q.get("answer_text"):
                        q["answer_text"] = q["answer_text"].replace(original, corrected)
                    break
                
    # Next, build the final processed_questions list combining the new text and validation status
    correction_map = {c.question_number: c for c in corrections}
    for q in corrected_blocks:
        q_num = q.get("question_number", 0)
        c = correction_map.get(q_num)
        
        marked = q.get("marked_correct_answer") or q.get("marked_answer")
        orig_explanation = q.get("explanation") or q.get("answer_text") or ""

        # Pop any masked base64 images extracted in _build_page_prompt
        extracted_images_stem = q.pop("_extracted_images_stem", {})
        extracted_images_exp = q.pop("_extracted_images_exp", {})
        q.pop("_clean_stem", None)
        q.pop("_clean_exp", None)

        from ..ingest.math_normalizer import normalize_math_in_text

        # Apply LLM stem math correction if provided, else restore owned images
        cand_stem = getattr(c, "corrected_stem", None)
        if cand_stem and str(cand_stem).strip():
            cand_stem_clean = str(cand_stem).strip()
            cand_stem_clean = _restore_owned_images(cand_stem_clean, extracted_images_stem, q.get("stem", ""))
            q["stem"] = normalize_math_in_text(cand_stem_clean)
        else:
            stem_val = _restore_owned_images(q.get("stem", ""), extracted_images_stem, q.get("stem", ""))
            q["stem"] = normalize_math_in_text(stem_val)

        # Apply LLM options math correction if provided, else normalize existing options
        cand_opts = getattr(c, "corrected_options", None)
        if cand_opts is not None and isinstance(cand_opts, dict):
            if len(cand_opts) > 0:
                q["options"] = {k: normalize_math_in_text(str(v)) for k, v in cand_opts.items()}
            else:
                q["options"] = {}
        elif "options" in q and isinstance(q["options"], dict):
            for ok, ov in q["options"].items():
                if isinstance(ov, str):
                    q["options"][ok] = normalize_math_in_text(ov)

        # Determine corrected explanation
        final_explanation = None
        has_corrected_exp = False
        if c and getattr(c, "corrected_explanation", None) and str(c.corrected_explanation).strip():
            final_explanation = _restore_owned_images(
                str(c.corrected_explanation).strip(), extracted_images_exp, orig_explanation
            )
            has_corrected_exp = True
        elif orig_explanation:
            final_explanation = orig_explanation

        if final_explanation:
            final_explanation = normalize_math_in_text(final_explanation)
            q["explanation"] = final_explanation
            q["answer_text"] = final_explanation

        if c:
            from ..ingest.option_answer_separator import classify_question_type, is_valid_subparts
            cur_opts = q.get("options", {})
            llm_qtype = getattr(c, "question_type", None)
            llm_type_str = str(llm_qtype).strip().lower() if llm_qtype else ""

            # Respect LLM question validation: never force descriptive into MCQ and never label answers as subquestions
            if llm_type_str in ("descriptive_single", "descriptive_subquestions"):
                q["options"] = {}
                q["marked_answer"] = None
                q["marked_correct_answer"] = None
                marked = None
                # Never copy answer options into sub_parts!
                # Only populate sub_parts if the question stem itself has genuine directive subparts
                if q.get("stem"):
                    from ..ingest.option_answer_separator import extract_subparts_from_text, is_valid_subparts
                    extracted_sub = extract_subparts_from_text(q.get("stem", ""))
                    if extracted_sub and is_valid_subparts(extracted_sub):
                        q["sub_parts"] = extracted_sub
                        q["question_type"] = "descriptive_subquestions"
                    else:
                        q["sub_parts"] = None
                        q["question_type"] = "descriptive_single"
                else:
                    q["sub_parts"] = None
                    q["question_type"] = "descriptive_single"
            elif cur_opts and len(cur_opts) >= 2:
                det_type, det_sub = classify_question_type(cur_opts, final_explanation, marked)
                if det_type == "mcq":
                    q["question_type"] = "mcq"
                elif det_sub and is_valid_subparts(det_sub):
                    q["question_type"] = "descriptive_subquestions"
                    q["sub_parts"] = det_sub
                    q["options"] = {}
                    q["marked_answer"] = None
                    q["marked_correct_answer"] = None
                    marked = None
                else:
                    q["question_type"] = "descriptive_single"
                    q["sub_parts"] = None
                    q["options"] = {}
                    q["marked_answer"] = None
                    q["marked_correct_answer"] = None
                    marked = None
            else:
                if llm_type_str in ("mcq", "descriptive_single", "descriptive_subquestions"):
                    verified_type = llm_type_str
                    q["question_type"] = verified_type
                    if verified_type == "mcq" and not q.get("options"):
                        from ..graph.nodes import separate_stem_and_options
                        re_stem, re_opts, re_ans, re_exp = separate_stem_and_options(q.get("stem", ""))
                        if re_opts and len(re_opts) >= 2:
                            q["stem"] = re_stem
                            q["options"] = re_opts
                            if not marked and re_ans:
                                marked = re_ans
                                q["marked_answer"] = re_ans
                                q["marked_correct_answer"] = re_ans
                    elif verified_type.startswith("descriptive"):
                        q["options"] = {}
                        q["marked_answer"] = None
                        q["marked_correct_answer"] = None
                        marked = None
                        if not q.get("sub_parts") and q.get("stem"):
                            from ..ingest.option_answer_separator import extract_subparts_from_text
                            extracted_sub = extract_subparts_from_text(q.get("stem", ""))
                            if extracted_sub and is_valid_subparts(extracted_sub):
                                q["sub_parts"] = extracted_sub
                                first_key = sorted(extracted_sub.keys())[0]
                                fk_esc = re.escape(first_key)
                                first_sub_pattern = re.compile(
                                    rf'(?:(?:^|\n|\s{{2,}}|\.\s+)\([{fk_esc}{fk_esc.upper()}]\)\s*)|'
                                    rf'(?:(?:^|\n|[ \t]{{2,}}|\.\s+)[{fk_esc}{fk_esc.upper()}][\.\)\:-]\s*)'
                                )
                                m_sub = first_sub_pattern.search(q["stem"])
                                if m_sub:
                                    intro = q["stem"][:m_sub.start()].strip()
                                    if re.match(r"^(?:question\s*\d*|q\d*|ചോദ്യം\s*\d*|[\d\.\:\-\s])*$", intro, re.IGNORECASE):
                                        q["stem"] = ""
                                    else:
                                        q["stem"] = intro

            current_qtype = q.get("question_type", "unknown")
            if current_qtype.startswith("descriptive"):
                validated_answer = None
                is_valid = True
                reasoning = final_explanation or "Descriptive question — solution verified."
                llm_status = "completed"
                llm_details = f"{provider.title()} verified descriptive question."
                if has_corrected_exp:
                    llm_details += " Corrected solution explanation."
            elif not ans_val_on:
                validated_answer = None
                is_valid = False
                reasoning = final_explanation or "Answer validation disabled in settings (.env)"
                llm_status = "skipped"
                llm_details = "Answer validation skipped (disabled in settings)"
            else:
                c_val = getattr(c, "validated_answer", None)
                if c_val and str(c_val).strip():
                    m_letter = re.search(r'\b([A-D])\b', str(c_val).upper())
                    validated_answer = m_letter.group(1) if m_letter else str(c_val).strip().upper()[:1]
                else:
                    validated_answer = marked if c.answer_status == "agrees" else None

                is_valid = bool(validated_answer and marked and validated_answer.upper() == marked.upper())
                reasoning = final_explanation or ("Answer key verified and matches" if is_valid else (f"AI verified Option {validated_answer} (differs from marked {marked})" if validated_answer else "Answer disagreement detected"))
                llm_status = "completed"
                if is_valid:
                    llm_details = f"{provider.title()} verified answer (Agreed: Option {validated_answer})"
                else:
                    llm_details = f"{provider.title()} disagreed: verified Option {validated_answer or 'different'} vs marked Option {marked}"
                if has_corrected_exp:
                    llm_details += " and corrected solution explanation."
                else:
                    llm_details += "."
        else:
            validated_answer = None
            is_valid = False
            reasoning = final_explanation or "LLM call skipped or offline"
            llm_status = "skipped"
            llm_details = "LLM validation skipped / offline"
        
        # Build step-by-step pipeline status
        has_math_img = "<img" in q.get("stem", "") or any("<img" in str(v) for v in q.get("options", {}).values())
        has_latex = "$" in q.get("stem", "") or "\\frac" in q.get("stem", "")

        q_corrections = [it for it in accepted if it.get("question_number") == q_num]
        if not word_corr_on:
            cache_status = "skipped"
            cache_details = "Word correction disabled in .env (ENABLE_WORD_CORRECTION=false)"
        else:
            cache_status = "completed"
            cache_details = f"{len(q_corrections)} word correction(s) applied" if q_corrections else "Passed dictionary & font cache validation"

        ocr_solved = q.pop("_ocr_solved_result", None)
        if ocr_solved:
            validated_answer = ocr_solved.get("validated_answer")
            is_valid = ocr_solved.get("is_valid", False)
            if not final_explanation:
                reasoning = ocr_solved.get("reasoning", "OCR fallback applied")
            llm_status = "completed"
            llm_details = f"Re-verified after OCR: {ocr_solved.get('reasoning', '')}"

        pipeline_steps = {
            "scanned": {
                "name": "scanned",
                "label": "Scanned",
                "status": "completed",
                "details": f"Extracted from PDF Page {(q.get('page_number') or 0) + 1}"
            },
            "checked_with_caches": {
                "name": "checked_with_caches",
                "label": "Checked with Caches",
                "status": cache_status,
                "details": cache_details
            },
            "send_errors_to_llms": {
                "name": "send_errors_to_llms",
                "label": "Send Errors to LLMs",
                "status": llm_status,
                "details": llm_details
            },
            "ocr_scanned": {
                "name": "ocr_scanned",
                "label": "OCR Scanned",
                "status": "completed" if (has_math_img or has_latex or ocr_solved) else "not_needed",
                "details": "Dynamic OCR fallback triggered by LLM" if ocr_solved else ("Math equation clips OCR parsed to LaTeX" if has_math_img else ("Math expressions parsed" if has_latex else "Text-only question (no image OCR needed)"))
            }
        }

        p_item = {
            **q,
            "marked_correct_answer": marked,
            "marked_answer": marked,
            "validated_answer": validated_answer,
            "is_valid": is_valid,
            "confidence": "high",  # We can just say high if it succeeded
            "explanation": final_explanation,
            "answer_text": final_explanation,
            "reasoning": reasoning,
            "used_web_search": False,
            "pipeline_steps": pipeline_steps
        }
        clean_question_options_and_answer(p_item)
        processed.append(p_item)

    logger.info(
        "[CorrectionNode] Done: %d corrections accepted, %d flags raised, %d cache promotions",
        len(accepted), len(review_flags), promoted,
    )
    
    # Compute token usage and summary
    existing_usage = state.get("token_usage", {})
    total_p = existing_usage.get("prompt_tokens", 0) + llm_p_tokens + fallback_p_tokens
    total_c = existing_usage.get("completion_tokens", 0) + llm_c_tokens + fallback_c_tokens
    total_tokens = total_p + total_c
    cost_usd = calculate_token_cost(total_p, total_c, provider)

    updated_token_usage = {
        "prompt_tokens": total_p,
        "completion_tokens": total_c,
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost_usd,
    }

    gemini_on = provider == "gemini"
    summary = {
        "total": len(processed),
        "validated": sum(1 for p in processed if p.get("is_valid")),
        "flagged": sum(1 for p in processed if (p.get("validated_answer") and not p.get("is_valid"))),
        "unvalidated": sum(1 for p in processed if not p.get("validated_answer")),
        "gemini_enabled": gemini_on,
        "answer_validation_enabled": True,
        "token_usage": updated_token_usage,
    }

    return {
        "extracted_blocks": corrected_blocks,
        "processed_questions": processed,
        "summary": summary,
        "token_usage": updated_token_usage,
        "flagged_words": all_flagged,
        "correction_results": [
            {"question_number": c.question_number, "answer_status": c.answer_status}
            for c in corrections
        ],
        "review_flags": [
            {"question_number": f["question_number"], "flag_type": f["flag_type"]}
            for f in review_flags
        ]
    }


# ---------------------------------------------------------------------------
# On-Demand Single Question Rewrite (Spec §10)
# ---------------------------------------------------------------------------

async def fix_single_question_with_llm(q_data: dict, provider: str | None = None) -> dict:
    """
    Called via API endpoint when a user explicitly clicks "Rewrite with AI".
    Sends the single question directly to the LLM for a full rewrite.
    """
    if provider is None or provider == "none":
        provider = get_on_demand_llm_provider()

    from ..ingest.option_answer_separator import clean_question_options_and_answer, separate_options_dict, classify_question_type
    clean_question_options_and_answer(q_data)

    q_num = q_data.get("question_number", 0)
    stem = q_data.get("stem", "")
    options = q_data.get("options", {})
    marked = q_data.get("marked_correct_answer") or q_data.get("marked_answer") or "Unknown"
    raw_exp = q_data.get("explanation") or q_data.get("answer_text") or q_data.get("reasoning") or ""
    explanation = _clean_explanation_text(raw_exp)

    # Send stable, field-owned image markers instead of image bytes to the LLM.
    clean_stem_for_prompt, extracted_images_stem = _mask_owned_images(stem, f"q{q_num}:stem")
    clean_exp_for_prompt, extracted_images_exp = _mask_owned_images(explanation, f"q{q_num}:explanation")

    prompt = f"--- Question {q_num} ---\nStem: {clean_stem_for_prompt}\n"
    if options:
        prompt += "Options:\n"
        for key in sorted(options.keys()):
            prompt += f"  ({key}) {options[key]}\n"
    prompt += f"Marked correct answer: {marked}\n"
    if clean_exp_for_prompt:
        prompt += f"Explanation from PDF: {clean_exp_for_prompt}\n"
    prompt += "\nReview typos (Malayalam/English), question type, LaTeX equations, and format paragraphs for readability."

    corrections = None
    p_tok = 0
    c_tok = 0

    async def _try_groq():
        nonlocal corrections, p_tok, c_tok
        api_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if api_key and not api_key.startswith("your_"):
            try:
                from groq import AsyncGroq
                client = AsyncGroq(api_key=api_key)
                response = await client.chat.completions.create(
                    model=GROQ_CORRECTION_MODEL,
                    messages=[
                        {"role": "system", "content": DIRECT_CORRECTION_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=4096,
                )
                if hasattr(response, "usage") and response.usage:
                    p_tok = response.usage.prompt_tokens or 0
                    c_tok = response.usage.completion_tokens or 0
                raw_text = response.choices[0].message.content or ""
                cleaned = _clean_json_output(raw_text)
                parsed = DirectPageCorrectionResponse.model_validate_json(cleaned)
                corrections = parsed.corrections
            except Exception as e:
                logger.error("[CorrectionNode] Groq single-q rewrite failed: %s", e)

    async def _try_gemini():
        nonlocal corrections, p_tok, c_tok
        api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
        if api_key and not api_key.startswith("your_"):
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=GEMINI_CORRECTION_MODEL,
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                    config=types.GenerateContentConfig(
                        system_instruction=DIRECT_CORRECTION_SYSTEM_PROMPT,
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                )
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    p_tok = response.usage_metadata.prompt_token_count or 0
                    c_tok = response.usage_metadata.candidates_token_count or 0
                raw_text = response.text
                if raw_text:
                    cleaned = _clean_json_output(raw_text)
                    parsed = DirectPageCorrectionResponse.model_validate_json(cleaned)
                    corrections = parsed.corrections
            except Exception as e:
                logger.error("[CorrectionNode] Gemini single-q rewrite failed: %s", e)

    if provider == "groq":
        await _try_groq()
        if not corrections:
            logger.warning("[CorrectionNode] Groq failed; falling back to Gemini for single-q rewrite")
            await _try_gemini()
    elif provider == "gemini":
        await _try_gemini()
        if not corrections:
            logger.warning("[CorrectionNode] Gemini failed or quota exhausted; falling back to Groq for single-q rewrite")
            await _try_groq()

    if corrections and len(corrections) > 0:
        c = corrections[0]
        cost_usd = calculate_token_cost(p_tok, c_tok, provider)

        # Clean any accidental question number prefix in stem
        clean_stem = c.corrected_stem.strip()
        clean_stem = re.sub(r'^(?:Question\s*|Q\.?\s*|ചോദ്യം\s*)\d+\s*(?::\-?|[\.\):-])\s*', '', clean_stem, flags=re.IGNORECASE)
        clean_stem = re.sub(r'^\d+\s*[\.\):-]\s+', '', clean_stem)
        clean_stem = re.sub(r'(?i)\s*\[\s*\d+(?:\.\d+)?(?:\s*/\s*\d+)?\s*(?:marks?|mark|m)\s*\]', '', clean_stem)
        clean_stem = re.sub(r'(?i)\s*\(\s*\d+(?:\.\d+)?(?:\s*/\s*\d+)?\s*(?:marks?|mark|m)\s*\)', '', clean_stem)
        from ..ingest.math_normalizer import normalize_math_in_text
        clean_stem = normalize_math_in_text(clean_stem)

        clean_stem = _restore_owned_images(clean_stem, extracted_images_stem, stem)

        # Clean any accidental option prefix in options and separate any trailing answer/solution
        # CRITICAL: Must be in parens/brackets or followed by delimiter/space.
        # NEVER strip 'B' from 'Both' or 'A' from 'All'!
        clean_opts = {}
        for k, v in c.corrected_options.items():
            if isinstance(v, str):
                k_esc = re.escape(k)
                pat = rf"^\s*(?:\([\s]*[{k_esc}{k_esc.lower()}][\s]*\)[\.\:\-]?|\[[\s]*[{k_esc}{k_esc.lower()}][\s]*\][\.\:\-]?|[{k_esc}{k_esc.lower()}][\.\)\:\-]\s*|[{k_esc}{k_esc.lower()}](?=\s+))\s*"
                v_clean = re.sub(pat, '', v.strip())
                v_clean = re.sub(r'\bOth\b', 'Both', v_clean)
                v_clean = re.sub(r'^(?:[oO]th)\b', 'Both', v_clean)
                def _fix_both_cn(m: re.Match) -> str:
                    idx = m.start()
                    if idx == 0 or re.search(r'(?:^|[.!?\n]\s*)$', v_clean[:idx]):
                        return 'Both'
                    return 'both'
                v_clean = re.sub(r'\both\b', _fix_both_cn, v_clean)
                v_clean = re.sub(r'\bll\s+of\s+the\s+above\b', 'All of the above', v_clean, flags=re.IGNORECASE)
                clean_opts[k] = normalize_math_in_text(v_clean)
            else:
                clean_opts[k] = str(v)

        c_val = getattr(c, "validated_answer", None)
        if c_val and str(c_val).strip():
            m_letter = re.search(r'\b([A-D])\b', str(c_val).upper())
            val_ans = m_letter.group(1) if m_letter else str(c_val).strip().upper()[:1]
        else:
            val_ans = marked if c.answer_status == "agrees" else None

        # Use only the PDF explanation, corrected by the LLM when available.
        reasoning = getattr(c, "corrected_explanation", None) or clean_exp_for_prompt
        if reasoning:
            reasoning = re.sub(r'^\(?([A-D])\)?[\s\.\:\-]+', '', reasoning).strip()
            MAL_CHAR = r'[\u0D00-\u0D7F]'
            reasoning = re.sub(rf'(?<={MAL_CHAR})[~}}]+|(?<=[~}}]){MAL_CHAR}', '', reasoning)
            reasoning = normalize_math_in_text(reasoning)

        reasoning = _restore_owned_images(reasoning, extracted_images_exp, raw_exp)

        clean_opts, sep_ans, sep_exp = separate_options_dict(clean_opts, val_ans, reasoning)
        if sep_ans and not val_ans:
            val_ans = sep_ans
        if sep_exp and not reasoning:
            reasoning = sep_exp
        
        from ..ingest.option_answer_separator import heal_assertion_reason_question
        clean_stem, clean_opts = heal_assertion_reason_question(clean_stem, clean_opts)
        
        sub_p = None
        llm_qtype = getattr(c, "question_type", None)
        llm_type_str = str(llm_qtype).strip().lower() if llm_qtype else ""

        # Respect LLM question validation: never force descriptive questions into MCQs and never label answers as subquestions
        if llm_type_str in ("descriptive_single", "descriptive_subquestions"):
            clean_opts = {}
            sub_p = None
            val_ans = None
            cur_marked = None
            q_type = "descriptive_single"
            if clean_stem:
                from ..ingest.option_answer_separator import extract_subparts_from_text, is_valid_subparts
                extracted_sub = extract_subparts_from_text(clean_stem)
                if extracted_sub and is_valid_subparts(extracted_sub):
                    sub_p = extracted_sub
                    q_type = "descriptive_subquestions"
        elif clean_opts and len(clean_opts) >= 2:
            det_type, det_sub = classify_question_type(clean_opts, reasoning, val_ans or marked)
            if det_type == "mcq":
                q_type = "mcq"
                sub_p = None
            elif det_sub:
                q_type = det_type
                sub_p = det_sub
            else:
                q_type = "descriptive_single"
                sub_p = None
                clean_opts = {}
        else:
            if not llm_type_str:
                q_type, sub_p = classify_question_type(clean_opts, reasoning, val_ans or marked)
            else:
                q_type = llm_type_str
                _, sub_p = classify_question_type(clean_opts, reasoning, val_ans or marked)
                if not sub_p and clean_stem:
                    from ..ingest.option_answer_separator import extract_subparts_from_text, is_valid_subparts
                    extracted_sub = extract_subparts_from_text(clean_stem)
                    if extracted_sub and is_valid_subparts(extracted_sub):
                        sub_p = extracted_sub

        if q_type == "mcq" and clean_opts:
            final_opts = clean_opts
            cur_marked = marked if marked != "Unknown" else val_ans
            is_valid = bool(val_ans and cur_marked and val_ans.upper() == str(cur_marked).upper())
        elif sub_p:
            q_type = "descriptive_subquestions"
            final_opts = {}
            val_ans = None
            cur_marked = None
            is_valid = True
            if not reasoning:
                reasoning = "Descriptive question — solution verified"
            if clean_stem:
                first_key = sorted(sub_p.keys())[0]
                fk_esc = re.escape(first_key)
                first_sub_pattern = re.compile(
                    rf'(?:(?:^|\n|\s{{2,}}|\.\s+)\([{fk_esc}{fk_esc.upper()}]\)\s*)|'
                    rf'(?:(?:^|\n|[ \t]{{2,}}|\.\s+)[{fk_esc}{fk_esc.upper()}][\.\)\:-]\s*)'
                )
                m_sub = first_sub_pattern.search(clean_stem)
                if m_sub:
                    intro = clean_stem[:m_sub.start()].strip()
                    if re.match(r"^(?:question\s*\d*|q\d*|ചോദ്യം\s*\d*|[\d\.\:\-\s])*$", intro, re.IGNORECASE):
                        clean_stem = ""
                    else:
                        clean_stem = intro
        else:
            q_type = "descriptive_single"
            final_opts = {}
            val_ans = None
            cur_marked = None
            is_valid = True
            if not reasoning:
                reasoning = "Descriptive question — solution verified"

        res_dict = {
            "success": True,
            "corrected_stem": clean_stem,
            "corrected_options": final_opts,
            "question_type": q_type,
            "validated_answer": val_ans,
            "is_valid": is_valid,
            "reasoning": reasoning,
            "explanation": reasoning,
            "answer_text": reasoning,
            "corrected_explanation": reasoning,
            "token_usage": {
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": p_tok + c_tok,
                "estimated_cost_usd": cost_usd,
            }
        }
        if sub_p:
            res_dict["sub_parts"] = sub_p
        return res_dict
    return {"success": False, "error": "LLM failed to return a valid correction"}
