"""
word_router.py

Per-word routing decision module (spec §3).

Purpose: decide which words are worth spending an LLM call on.  This is
the free filter that keeps LLM volume proportional to actual problems
instead of the whole document.

Two independent signals feed the same routing decision:
1. Dictionary miss — word not found in the wordlist.
2. Font-mapper confidence — topo_sig (nearest-neighbor) resolution is
   inherently less certain than exact_hash and should route to the LLM
   regardless of whether the resulting word looks like a valid dictionary
   word.

A word routes to the LLM if EITHER signal fires, and only after a cache
miss — cache hits never reach the dictionary check at all.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

from .malayalam_validator import (
    MALAYALAM_RANGE,
    find_stray_chars,
    is_numeric_or_symbolic,
    load_wordlist,
)
from .correction_cache import CorrectionCache


@dataclass
class FlaggedWord:
    """A word flagged for LLM review."""
    word: str
    position: int           # char offset in the question text
    reason: str             # "dict_miss" | "low_confidence_glyph" | "both" | "stray_char"
    context: str = ""       # surrounding sentence for the LLM prompt


@dataclass
class RoutingResult:
    """Per-question routing result."""
    question_number: int
    flagged_words: list[FlaggedWord]
    cache_applied: list[dict]  # corrections already applied from cache
    corrected_text: str        # text after cache corrections applied


_TOKEN_RE = re.compile(r"[^\s]+", re.UNICODE)
_LEADING_PUNCT = re.compile(r"^[^\w\u0D00-\u0D7F]+", re.UNICODE)
_TRAILING_PUNCT = re.compile(r"[^\w\u0D00-\u0D7F]+$", re.UNICODE)


def _extract_core_token(raw_token: str) -> tuple[str, str, str]:
    """Strip leading/trailing punctuation from a token.
    Returns (prefix, core, suffix)."""
    lead = _LEADING_PUNCT.match(raw_token)
    prefix = lead.group(0) if lead else ""
    core = raw_token[len(prefix):]
    trail = _TRAILING_PUNCT.search(core)
    suffix = trail.group(0) if trail else ""
    token = core[: len(core) - len(suffix)] if suffix else core
    return prefix, token, suffix


def _has_low_confidence_chars(
    word: str,
    char_confidences: dict | None,
) -> bool:
    """
    Check if any character in `word` was resolved via a nearest-neighbor
    (topological signature) match rather than an exact contour hash.

    A topo_sig match is inherently less certain — it should route to the
    LLM regardless of whether the resulting word happens to look like a
    valid dictionary word.
    """
    if not char_confidences:
        return False

    for ch in word:
        code = ord(ch)
        # Check by character code or string representation
        conf = (
            char_confidences.get(code)
            or char_confidences.get(str(code))
            or char_confidences.get(ch)
        )
        if conf and conf != "exact_hash":
            return True

    return False


def _get_context(text: str, position: int, window: int = 80) -> str:
    """Extract surrounding context for a word at a given position."""
    start = max(0, position - window)
    end = min(len(text), position + window)
    return text[start:end]


def route_words_for_question(
    question_number: int,
    text: str,
    page_font_hashes: dict[str, str],
    char_confidences: dict | None,
    cache: CorrectionCache,
    document_id: str | None = None,
) -> RoutingResult:
    """
    Route words in a single question's text through the decision tree:

    1. Cache hit? → apply cached correction, done
    2. Font-mapper confidence low (nearest-neighbor)? → flag for LLM
    3. Dictionary miss? → flag for LLM
    4. None of the above → leave as-is

    Returns a RoutingResult with flagged words and cache-applied corrections.
    """
    wordlist = load_wordlist()

    # Step 1: Apply cache corrections to the whole text first
    corrected_text, cache_applied = cache.apply_corrections(
        text, page_font_hashes, document_id
    )

    # Step 2: Route remaining words
    flagged: list[FlaggedWord] = []

    # Merge all font-level confidence maps into a single lookup
    merged_conf: dict = {}
    if char_confidences:
        for font_conf in char_confidences.values():
            if isinstance(font_conf, dict):
                merged_conf.update(font_conf)

    for m in _TOKEN_RE.finditer(corrected_text):
        raw_token = m.group(0)
        offset = m.start()

        _, token, _ = _extract_core_token(raw_token)
        if not token:
            continue

        token_nfc = unicodedata.normalize("NFC", token)

        # Skip non-Malayalam tokens
        if is_numeric_or_symbolic(token_nfc):
            continue
        has_malayalam = any(
            MALAYALAM_RANGE[0] <= ord(c) <= MALAYALAM_RANGE[1]
            for c in token_nfc
        )
        if not has_malayalam:
            continue

        # Check for stray non-Malayalam characters
        stray = find_stray_chars(token_nfc)
        if stray:
            flagged.append(FlaggedWord(
                word=raw_token,
                position=offset,
                reason="stray_char",
                context=_get_context(corrected_text, offset),
            ))
            continue

        # Check both routing signals
        low_conf = _has_low_confidence_chars(token_nfc, merged_conf)
        dict_miss = token_nfc not in wordlist

        if low_conf and dict_miss:
            reason = "both"
        elif low_conf:
            reason = "low_confidence_glyph"
        elif dict_miss:
            reason = "dict_miss"
        else:
            continue  # Word is fine — exact hash match and in dictionary

        flagged.append(FlaggedWord(
            word=raw_token,
            position=offset,
            reason=reason,
            context=_get_context(corrected_text, offset),
        ))

    return RoutingResult(
        question_number=question_number,
        flagged_words=flagged,
        cache_applied=cache_applied,
        corrected_text=corrected_text,
    )


def route_words_for_page(
    questions: list[dict],
    page_font_hashes: dict[str, str],
    char_confidences: dict | None,
    cache: CorrectionCache,
    document_id: str | None = None,
) -> list[RoutingResult]:
    """
    Route words for all questions on a page.

    Args:
        questions: list of question dicts with at least
                   'question_number', 'stem', 'options', 'marked_answer'
        page_font_hashes: {font_name: md5_hash} for this page
        char_confidences: per-font confidence maps from font_mapper
        cache: the correction cache instance
        document_id: job/document ID for cache audit trail

    Returns: list of RoutingResult, one per question
    """
    results = []
    for q in questions:
        q_num = q.get("question_number", 0)

        # Build full question text for routing
        parts = [q.get("stem", "")]
        options = q.get("options", {})
        for key in sorted(options.keys()):
            parts.append(f"{key}. {options[key]}")

        full_text = "\n".join(parts)

        result = route_words_for_question(
            question_number=q_num,
            text=full_text,
            page_font_hashes=page_font_hashes,
            char_confidences=char_confidences,
            cache=cache,
            document_id=document_id,
        )
        results.append(result)

    return results
