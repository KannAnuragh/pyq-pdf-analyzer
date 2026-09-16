"""
malayalam_validator.py

Word-level validation layer for Malayalam PDF extraction pipelines.

Purpose
-------
Provides a dictionary-based yes/no membership check and stray-character
detection for each word in extracted text.  This module is a TRIAGE layer:
it tells you per word "trust this" / "flag for LLM correction" so the
downstream batched LLM call (correction_node.py) only processes words
that actually need attention.

Design (per spec §3 — Dictionary gate)
---------------------------------------
The wordlist (~152K entries) is used strictly as a yes/no membership check.
The earlier bigram model and Viterbi segmenter are NOT part of this —
those made corrections themselves and had accuracy problems (false positives
on inflected forms, bad splits on loanwords).  A membership check used
only to decide "should the LLM look at this" doesn't carry that risk.

Integration point: called from text_cleaner.py's clean_indic_text() pipeline
and from word_router.py for per-word routing decisions.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

MALAYALAM_RANGE = (0x0D00, 0x0D7F)
ZW_CHARS = {"\u200c", "\u200d"}  # ZWNJ / ZWJ — legitimate inside conjuncts

WORDLIST_PATH = Path(__file__).with_name("malayalam_wordlist.txt")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def load_wordlist(path: Path = WORDLIST_PATH) -> frozenset[str]:
    words = path.read_text(encoding="utf-8").splitlines()
    return frozenset(unicodedata.normalize("NFC", w) for w in words if w)


# --------------------------------------------------------------------------- #
# Low-level char checks
# --------------------------------------------------------------------------- #

def is_malayalam_char(ch: str) -> bool:
    return MALAYALAM_RANGE[0] <= ord(ch) <= MALAYALAM_RANGE[1] or ch in ZW_CHARS


def find_stray_chars(word: str) -> list[int]:
    """
    Indices of characters inside a word that are neither Malayalam-range,
    zero-width joiners, nor common digits/punctuation legitimately embedded
    in exam text (e.g. '5,9,13'). Anything else mid-word is almost always
    an unmapped-glyph fallback byte (Case 1 / the stray '{' in your screenshot).
    """
    allowed_extra = set("0123456789.,%-/()")
    return [
        i for i, ch in enumerate(word)
        if not is_malayalam_char(ch) and ch not in allowed_extra
    ]


def is_numeric_or_symbolic(word: str) -> bool:
    """Pure numbers / number-like tokens ('2021', '20', '5,') aren't Malayalam
    words at all — don't run them through the lexicon, just wave them through."""
    return bool(re.fullmatch(r"[0-9.,%/()\-]+", word))


def is_in_dictionary(word: str) -> bool:
    """Check if a word (NFC-normalized) is in the Malayalam wordlist."""
    wordlist = load_wordlist()
    return unicodedata.normalize("NFC", word) in wordlist


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass
class Flag:
    span: str
    char_offset: int
    reason: str
    suggestion: str | None = None


@dataclass
class ValidationResult:
    repaired_text: str
    flags: list[Flag] = field(default_factory=list)


_TOKEN_RE = re.compile(r"[^\s]+", re.UNICODE)


# --------------------------------------------------------------------------- #
# Simplified validation (dictionary gate only, no Viterbi/bigram correction)
# --------------------------------------------------------------------------- #

def validate_and_repair(text: str) -> ValidationResult:
    """
    Runs the word-level gate over `text` (already through existing
    NFC-normalize + matra-reorder cleaning).

    This simplified version:
    - Detects stray non-Malayalam characters in words → flags them
    - Detects dictionary misses → flags them
    - Does NOT attempt correction (no Viterbi, no bigram) — corrections
      are handled by the downstream batched LLM call in correction_node.py

    Returns repaired text (unchanged — this gate only flags, never corrects)
    plus a flags list for downstream routing.
    """
    wordlist = load_wordlist()

    flags: list[Flag] = []

    LEADING_PUNCT = re.compile(r"^[^\w\u0D00-\u0D7F]+", re.UNICODE)
    TRAILING_PUNCT = re.compile(r"[^\w\u0D00-\u0D7F]+$", re.UNICODE)

    for m in _TOKEN_RE.finditer(text):
        raw_token = m.group(0)
        offset = m.start()

        lead = LEADING_PUNCT.match(raw_token)
        prefix = lead.group(0) if lead else ""
        core = raw_token[len(prefix):]
        trail = TRAILING_PUNCT.search(core)
        suffix = trail.group(0) if trail else ""
        token = core[: len(core) - len(suffix)] if suffix else core

        if not token:
            continue

        token_nfc = unicodedata.normalize("NFC", token)

        if is_numeric_or_symbolic(token_nfc):
            continue

        # Bypass tokens that don't contain any Malayalam characters
        has_malayalam = any(MALAYALAM_RANGE[0] <= ord(c) <= MALAYALAM_RANGE[1] for c in token_nfc)
        if not has_malayalam:
            continue

        stray = find_stray_chars(token_nfc)
        if stray:
            flags.append(Flag(raw_token, offset, "stray_non_malayalam_char"))
            continue

        if token_nfc in wordlist:
            continue

        # Word not in dictionary — flag for downstream LLM routing.
        # No correction attempted here (per spec §3: dictionary is a
        # routing decision, not a corrector).
        flags.append(Flag(raw_token, offset, reason="dictionary_miss"))

    return ValidationResult(repaired_text=text, flags=flags)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    samples = [
        "5, 9, 13, 17,... എന്ന സമാന്തര ശ്രേ{ണിയുടെ ആദ്യ 20 പദങ്ങളുടെ തുക എത്ര?",
        "ഇന്ത്യൻ പാർലമെന്റ് 2021 പ്രകാരമുള്ള നിയമങ്ങൾ",
        "താഴെപറയുന്നവയിൽ വെസലിൽപയോഗിക്കുന്ന ഹാന്റ് ടൂൾ അല്ലാത്തത് ഏത്?",
    ]
    for s in samples:
        result = validate_and_repair(s)
        print("INPUT :", s)
        print("OUTPUT:", result.repaired_text)
        for f in result.flags:
            print(f"  FLAG  [{f.reason}] '{f.span}' @ {f.char_offset}")
        print()
