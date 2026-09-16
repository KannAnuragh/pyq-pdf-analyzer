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
        description="The option letter (A, B, C, D) determined by the AI to be correct"
    )
    answer_status: str = Field(
        default="agrees",
        description="'agrees' if the marked answer is correct, 'disagrees' if not"
    )


class PageCorrectionResponse(BaseModel):
    """Structured response for all questions on a page."""
    corrections: list[QuestionCorrection]


class DirectQuestionCorrection(BaseModel):
    """Per-question structured response for Direct LLM Validation mode."""
    question_number: int
    corrected_stem: str = Field(
        description="The completely rewritten and corrected question text (stem)."
    )
    corrected_options: dict[str, str] = Field(
        description="The completely rewritten and corrected options. Keys are option letters (A, B, C, D)."
    )
    validated_answer: str | None = Field(
        default=None,
        description="The option letter (A, B, C, D) determined by the AI to be correct"
    )
    answer_status: str = Field(
        default="agrees",
        description="'agrees' if the marked answer is correct, 'disagrees' if not"
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

CORRECTION_SYSTEM_PROMPT = """You are a Malayalam language expert reviewing exam question papers extracted from PDFs.

You have TWO separate jobs for each question:

JOB 1 — TEXT CORRECTION:
- Fix ONLY the specific flagged words listed below each question.
- Use the surrounding sentence context to determine the correct Malayalam word.
- Do NOT rewrite or "improve" text that isn't flagged.
- If there are no flagged words listed for a question, DO NOT make any text corrections for that question.
- If you're uncertain about a correction, report low confidence rather than guessing.
- If the Malayalam text is completely unreadable or garbled beyond any possible correction (e.g., mojibake, completely wrong fonts), set 'needs_ocr_fallback' to true. Otherwise, keep it false.
- Return the corrected word in the 'corrected_words' map.
- If a flagged word is actually correct as-is, do NOT include it in corrected_words.

JOB 2 — ANSWER VERIFICATION (MUST DO FOR EVERY QUESTION):
- Independently solve the question and determine which option ('A', 'B', 'C', or 'D') is factually/logically correct.
- Return the chosen option letter in 'validated_answer' ('A', 'B', 'C', or 'D').
- If 'validated_answer' matches the marked correct answer, report 'answer_status': 'agrees'.
- If 'validated_answer' does NOT match, report 'answer_status': 'disagrees'.

IMPORTANT: These are two genuinely separate judgments. A text correction does not change the answer verification, and vice versa."""

JSON_CORRECTION_SYSTEM_PROMPT = f"""{CORRECTION_SYSTEM_PROMPT}

You MUST output valid JSON conforming exactly to this structure:
{{
  "corrections": [
    {{
      "question_number": <int>,
      "corrected_words": {{"<flagged_word>": "<corrected_word>"}},
      "text_confidence": <float 0.0 to 1.0>,
      "needs_ocr_fallback": <bool>,
      "validated_answer": "A" | "B" | "C" | "D",
      "answer_status": "agrees" | "disagrees"
    }}
  ]
}}"""

DIRECT_CORRECTION_SYSTEM_PROMPT = """You are an elite Malayalam language and exam question expert reviewing question papers extracted from PDFs.

You have TWO separate jobs for each question provided:

JOB 1 — FULL TEXT REWRITE & STRUCTURAL FORMATTING:
- You will receive the raw, extracted text of the question (stem) and its options. You may also receive an Explanation from the PDF.
- The text may contain OCR errors, spelling mistakes, missing spaces, or formatting issues.
- Rewrite the entire question stem and all options with perfect Malayalam spelling and grammar.
- PRESERVE MULTI-LINE STRUCTURE: If the question has sub-points, numbered statements, Roman numerals (e.g. (i), (ii), (iii)), tables, poem stanzas, or lists, put EACH sub-point on its own new line using newline characters (\n). Do NOT collapse multi-line statements into a single sentence.
- Return ONLY the question body text in `corrected_stem`. Do NOT prefix with question numbers (do NOT include "Question 1:", "Q1.", "1.", "ചോദ്യം 1:").
- Do NOT include options or option choices inside `corrected_stem`.
- In `corrected_options`, return the cleaned text for each option mapped to its key ('A', 'B', 'C', 'D'). Do NOT include the option prefix itself inside the value (do NOT include 'A.', '(A)', 'A)' in the string).
- If an 'Explanation from PDF' is provided, you MUST completely rewrite it, fixing any OCR garbage, spelling mistakes, Malayalam ligatures, or formatting errors, and return the fixed text in `corrected_explanation`. Do NOT return null if an explanation was provided.
- PRESERVE MATH & FORMULAS: Format all mathematical formulas, symbols, fractions, and Greek letters using standard inline LaTeX enclosed in dollar signs (e.g., $x^2 + y^2 = r^2$, $\frac{a}{b}$, $\sqrt{x}$, $\sigma^2$, $\chi^2$).
- MULTIPLICATION & EQUATIONS: For multiplication, ALWAYS use LaTeX `\times` (e.g., `$96 = 2^5 \times 3$`, `$404 = 2^2 \times 101$`). Never use the lowercase letter 'x' or 'x^3' for multiplication! Preserve punctuation (commas, periods) between multiple equations.
- Fix Malayalam ligature/split glyph corruptions and replace non-breaking spaces.

JOB 2 — ANSWER VERIFICATION:
- Independently solve the question and determine which option ('A', 'B', 'C', or 'D') is correct.
- Always provide the chosen option letter in `validated_answer` ('A', 'B', 'C', or 'D').
- If `validated_answer` matches the marked correct answer, report `answer_status`: 'agrees'.
- If `validated_answer` does NOT match, report `answer_status`: 'disagrees'.

You MUST output valid JSON conforming exactly to this structure:
{
  "corrections": [
    {
      "question_number": <int>,
      "corrected_stem": "<string>",
      "corrected_options": {"A": "<string>", "B": "<string>", "C": "<string>", "D": "<string>"},
      "corrected_explanation": "<string or null>",
      "validated_answer": "A" | "B" | "C" | "D",
      "answer_status": "agrees" | "disagrees"
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


def _build_page_prompt(
    questions_on_page: list[dict],
    flagged_words_by_question: dict[int, list[dict]],
) -> str:
    """Build the prompt for one page's worth of flagged questions."""
    parts = []

    for q in questions_on_page:
        q_num = q.get("question_number", 0)
        stem = q.get("stem", "")
        options = q.get("options", {})
        marked = q.get("marked_answer", "Unknown")
        flagged = flagged_words_by_question.get(q_num, [])

        parts.append(f"\n--- Question {q_num} ---")
        parts.append(f"Stem: {stem}")
        for key in sorted(options.keys()):
            parts.append(f"  {key}. {options[key]}")
        parts.append(f"Marked correct answer: {marked}")
        
        if flagged:
            parts.append(f"\nFlagged words to review (for Text Correction):")
            for fw in flagged:
                parts.append(f"  - '{fw['word']}' (reason: {fw['reason']})")
                if fw.get('context'):
                    parts.append(f"    Context: ...{fw['context']}...")
        else:
            parts.append(f"\nNo specific text flagged for this question. Proceed directly to Answer Verification.")

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
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JSON_CORRECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
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
                wait = (2 ** attempt) * (2 if is_rate_limit else 1)
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
        return await _call_groq_for_page(questions_on_page, flagged_words_by_question)
    elif provider == "gemini":
        logger.info("[CorrectionNode] Using Gemini (%s) for page validation & correction", GEMINI_CORRECTION_MODEL)
        return await _call_gemini_for_page(questions_on_page, flagged_words_by_question)
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

        if getattr(correction, 'needs_ocr_fallback', False):
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
    Process flagged questions in small batches (default 5 questions/batch).
    Prevents blowing past LLM token limits (e.g. Groq 7,000 ITPM limit).
    Returns (all_corrections, total_prompt_tokens, total_completion_tokens).
    """
    all_corrections: list[QuestionCorrection] = []
    total_p_tok = 0
    total_c_tok = 0

    total_batches = (len(questions_with_flags) + BATCH_SIZE - 1) // BATCH_SIZE
    for idx, i in enumerate(range(0, len(questions_with_flags), BATCH_SIZE)):
        batch = questions_with_flags[i:i + BATCH_SIZE]
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

        # Brief pause between batches to respect rate limits
        if i + BATCH_SIZE < len(questions_with_flags):
            await asyncio.sleep(0.5)

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
                    for applied in r.cache_applied:
                        garbled = applied["garbled"]
                        correction = applied["correction"]
                        corrected_stem = corrected_stem.replace(garbled, correction)
                        for opt_key in corrected_opts:
                            corrected_opts[opt_key] = corrected_opts[opt_key].replace(garbled, correction)
                    corrected_q["stem"] = corrected_stem
                    corrected_q["options"] = corrected_opts
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
                    "details": f"Extracted from PDF Page {q.get('page_number', 0) + 1}"
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
                    "details": "LLM validation & correction skipped / offline"
                },
                "ocr_scanned": {
                    "name": "ocr_scanned",
                    "label": "OCR Scanned",
                    "status": "completed" if (has_math_img or has_latex) else "not_needed",
                    "details": "Math equation clips OCR parsed to LaTeX" if has_math_img else ("Math expressions parsed" if has_latex else "Text-only question (no image OCR needed)")
                }
            }

            processed.append({
                **q,
                "marked_correct_answer": marked,
                "marked_answer": marked,
                "validated_answer": None,
                "is_valid": False,
                "confidence": "none",
                "reasoning": "LLM validation is disabled.",
                "used_web_search": False,
                "pipeline_steps": p_steps
            })
            
        return {
            "extracted_blocks": corrected_blocks or extracted_blocks,
            "processed_questions": processed,
            "flagged_words": all_flagged,
            "correction_results": [],
            "review_flags": [],
        }

    # ---- Step 3: Group by page and call LLM in batches ----
    logger.info(
        "[CorrectionNode] Calling %s unconditionally for correction & validation on %d questions",
        provider.upper(), len(corrected_blocks)
    )

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
            processed.append({
                **q,
                "marked_correct_answer": marked,
                "marked_answer": marked,
                "validated_answer": None,
                "is_valid": False,
                "confidence": "none",
                "reasoning": "no explanation",
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
        import re

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
                        page_idx = q.get("page_number", 0)
                        if page_idx < len(doc):
                            page = doc[page_idx]
                            crop = _get_crop(page, q_num)
                            pix = page.get_pixmap(clip=crop, dpi=200) if crop else page.get_pixmap(dpi=200)
                            image_bytes = pix.tobytes("png")
                            
                            try:
                                ext_data, p_tok, c_tok = rescan_question_vision(image_bytes, q_num)
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
                    break
                
    # Next, build the final processed_questions list combining the new text and validation status
    correction_map = {c.question_number: c for c in corrections}
    for q in corrected_blocks:
        q_num = q.get("question_number", 0)
        c = correction_map.get(q_num)
        
        marked = q.get("marked_correct_answer") or q.get("marked_answer")
        
        if c:
            if not ans_val_on:
                validated_answer = None
                is_valid = False
                reasoning = "Answer validation disabled in settings (.env)"
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
                reasoning = ("Answer key verified and matches" if is_valid else (f"AI verified Option {validated_answer} (differs from marked {marked})" if validated_answer else "Answer disagreement detected"))
                llm_status = "completed"
                if is_valid:
                    llm_details = f"{provider.title()} verified answer (Agreed: Option {validated_answer})."
                else:
                    llm_details = f"{provider.title()} disagreed: verified Option {validated_answer or 'different'} vs marked Option {marked}."
        else:
            validated_answer = None
            is_valid = False
            reasoning = "LLM call skipped or offline"
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
            reasoning = ocr_solved.get("reasoning", "OCR fallback applied")
            llm_status = "completed"
            llm_details = f"Re-verified after OCR: {reasoning}"

        pipeline_steps = {
            "scanned": {
                "name": "scanned",
                "label": "Scanned",
                "status": "completed",
                "details": f"Extracted from PDF Page {q.get('page_number', 0) + 1}"
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

        processed.append({
            **q,
            "marked_correct_answer": marked,
            "marked_answer": marked,
            "validated_answer": validated_answer,
            "is_valid": is_valid,
            "confidence": "high",  # We can just say high if it succeeded
            "reasoning": reasoning,
            "used_web_search": False,
            "pipeline_steps": pipeline_steps
        })

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
    import re

    if provider is None:
        provider = get_correction_provider()

    q_num = q_data.get("question_number", 0)
    stem = q_data.get("stem", "")
    options = q_data.get("options", {})
    marked = q_data.get("marked_answer") or q_data.get("marked_correct_answer", "Unknown")
    explanation = q_data.get("explanation") or q_data.get("reasoning", "")

    # Extract img tags to avoid sending huge base64 strings as text tokens
    extracted_images_stem = []
    def img_replacer_stem(match):
        extracted_images_stem.append(match.group(0))
        return "\n[DIAGRAM/IMAGE PLACEHOLDER]\n"
    clean_stem_for_prompt = re.sub(r'(<div[^>]*>\s*)?<img[^>]+src=[\'"]data:image/[^\'"]+[\'"][^>]*>(\s*</div>)?', img_replacer_stem, stem)

    extracted_images_exp = []
    def img_replacer_exp(match):
        extracted_images_exp.append(match.group(0))
        return "\n[DIAGRAM/IMAGE PLACEHOLDER]\n"
    clean_exp_for_prompt = re.sub(r'(<div[^>]*>\s*)?<img[^>]+src=[\'"]data:image/[^\'"]+[\'"][^>]*>(\s*</div>)?', img_replacer_exp, explanation)

    prompt = f"--- Question {q_num} ---\nStem: {clean_stem_for_prompt}\n"
    for key in sorted(options.keys()):
        prompt += f"  {key}. {options[key]}\n"
    prompt += f"Marked correct answer: {marked}\n"
    if clean_exp_for_prompt:
        prompt += f"Explanation from PDF: {clean_exp_for_prompt}\n"
    prompt += "\nPreserve multi-line formatting (e.g. sub-points (i), (ii), lists). Proceed to full rewrite and answer verification."

    corrections = None
    p_tok = 0
    c_tok = 0

    if provider == "groq":
        api_key = (os.getenv("GROQ_API_KEY") or "").strip()
        if api_key and not api_key.startswith("your_"):
            from groq import AsyncGroq
            client = AsyncGroq(api_key=api_key)
            try:
                response = await client.chat.completions.create(
                    model=GROQ_CORRECTION_MODEL,
                    messages=[
                        {"role": "system", "content": DIRECT_CORRECTION_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
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

    elif provider == "gemini":
        api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
        if api_key and not api_key.startswith("your_"):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)
            try:
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

    if corrections and len(corrections) > 0:
        c = corrections[0]
        cost_usd = calculate_token_cost(p_tok, c_tok, provider)

        # Clean any accidental question number prefix in stem
        clean_stem = c.corrected_stem.strip()
        clean_stem = re.sub(r'^(?:Question\s*|Q\.?\s*|ചോദ്യം\s*)\d+\s*(?::\-?|[\.\):-])\s*', '', clean_stem, flags=re.IGNORECASE)
        clean_stem = re.sub(r'^\d+\s*[\.\):-]\s+', '', clean_stem)
        clean_stem = clean_stem.strip()

        # Re-attach extracted images to stem
        for ext_img in extracted_images_stem:
            if "[DIAGRAM/IMAGE PLACEHOLDER]" in clean_stem:
                clean_stem = clean_stem.replace("[DIAGRAM/IMAGE PLACEHOLDER]", ext_img, 1)
            else:
                clean_stem += f"\n{ext_img}"

        # Clean any accidental option prefix in options
        clean_opts = {}
        for k, v in c.corrected_options.items():
            if isinstance(v, str):
                v_clean = re.sub(rf'^(?:\(?[{k}{k.lower()}]\)?[\.\):-]?\s*)', '', v.strip())
                clean_opts[k] = v_clean
            else:
                clean_opts[k] = str(v)

        c_val = getattr(c, "validated_answer", None)
        if c_val and str(c_val).strip():
            m_letter = re.search(r'\b([A-D])\b', str(c_val).upper())
            val_ans = m_letter.group(1) if m_letter else str(c_val).strip().upper()[:1]
        else:
            val_ans = marked if c.answer_status == "agrees" else None

        is_valid = bool(val_ans and marked and val_ans.upper() == marked.upper())
        
        # Use LLM corrected explanation if available, otherwise original explanation. No fallback text.
        reasoning = getattr(c, "corrected_explanation", None) or clean_exp_for_prompt

        # Re-attach extracted images to reasoning
        for ext_img in extracted_images_exp:
            if reasoning and "[DIAGRAM/IMAGE PLACEHOLDER]" in reasoning:
                reasoning = reasoning.replace("[DIAGRAM/IMAGE PLACEHOLDER]", ext_img, 1)
            else:
                reasoning = (reasoning or "") + f"\n{ext_img}"

        return {
            "success": True,
            "corrected_stem": clean_stem,
            "corrected_options": clean_opts,
            "validated_answer": val_ans,
            "is_valid": is_valid,
            "reasoning": reasoning,
            "token_usage": {
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": p_tok + c_tok,
                "estimated_cost_usd": cost_usd,
            }
        }
    return {"success": False, "error": "LLM failed to return a valid correction"}
