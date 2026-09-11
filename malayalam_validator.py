"""
malayalam_validator.py

Word-level validation/repair layer for Malayalam PDF extraction pipelines.

Purpose
-------
Your existing pipeline (contour-hash glyph mapping + sanity_check_malayalam_text)
validates at the PAGE level: "is >=85% of this page's characters valid Malayalam
Unicode?". That check passes even when individual words are corrupted, because a
handful of dropped/garbled characters barely moves a page-wide percentage. This
module adds a SECOND, WORD-level gate that catches exactly the three failure
modes seen in production:

  1. Stray non-Malayalam bytes injected mid-word (unmapped glyph fallback)
  2. Missing spaces at word boundaries (over-aggressive continuous-script guard)
  3. Dropped/garbled characters inside a word (glyph contour-hash miss)

Design constraints honored: 100% offline, no GPU, no network calls. The only
asset required is a plaintext Malayalam wordlist (bundled: malayalam_wordlist.txt,
~143k entries derived from the SMC/hunspell-ml community dictionary, ~5MB).

This is NOT a spellchecker and does not try to "auto-correct" prose. It is a
TRIAGE layer: it tells you, per word, "trust this" / "probably a merged pair,
here's the split" / "cannot confidently parse this — flag for OCR review",
so silent corruption stops reaching your output JSON.

Integration point: call `validate_and_repair()` as the last step inside
text_cleaner.py's clean_indic_text() pipeline, AFTER matra reordering and
BEFORE question/option parsing. Anything in the returned `flags` list should
be pushed into your existing OCR-fallback queue instead of shipping as-is.
"""

from __future__ import annotations

import math
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


@lru_cache(maxsize=1)
def build_bigram_model(path: Path = WORDLIST_PATH) -> dict[str, float]:
    """
    Character-bigram log-probabilities built from the wordlist itself.

    We don't have a large running-text corpus offline, so the wordlist doubles
    as the training corpus. This is intentionally crude — it is a BACKSTOP for
    words that aren't literal dictionary entries (Malayalam is agglutinative;
    142k entries will never cover every inflected/compound form), not a
    replacement for the dictionary check.
    """
    words = load_wordlist(path)
    counts: dict[str, int] = {}
    unigram: dict[str, int] = {}
    for w in words:
        chars = ["^"] + list(w) + ["$"]
        for a, b in zip(chars, chars[1:]):
            counts[a + b] = counts.get(a + b, 0) + 1
            unigram[a] = unigram.get(a, 0) + 1
    total_bigrams = sum(counts.values())
    vocab_size = len(unigram) + 1
    # Laplace-smoothed log-probabilities
    model: dict[str, float] = {}
    for bg, c in counts.items():
        a = bg[0]
        denom = unigram.get(a, 0) + vocab_size
        model[bg] = math.log((c + 1) / denom)
    model["__default__"] = math.log(1 / (total_bigrams + vocab_size))
    return model


def bigram_logprob(word: str, model: dict[str, float]) -> float:
    chars = ["^"] + list(word) + ["$"]
    default = model["__default__"]
    score = 0.0
    for a, b in zip(chars, chars[1:]):
        score += model.get(a + b, default)
    return score / max(len(chars) - 1, 1)  # normalize by length


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


# --------------------------------------------------------------------------- #
# Viterbi word segmentation — repairs missing / wrongly-merged spaces
# --------------------------------------------------------------------------- #

def viterbi_segment(run: str, wordlist: frozenset[str], bigram_model: dict[str, float],
                     max_word_len: int = 20) -> tuple[list[str], float] | None:
    """
    Best segmentation of a space-free character run into dictionary words,
    using dynamic programming (unigram-style, standard word-segmentation DP —
    same idea as segmenting 'thequickbrownfox'). Falls back to bigram score
    for spans not found in the dictionary so real-but-uncatalogued words
    don't block a valid split. Returns None if no split scores acceptably
    (signal: don't guess, flag it instead).
    """
    n = len(run)
    if n == 0:
        return [], 0.0
    NEG_INF = float("-inf")
    best_score = [NEG_INF] * (n + 1)
    best_split = [-1] * (n + 1)
    best_score[0] = 0.0

    SEGMENT_PENALTY = 4.0   # flat cost per segment: fewer, longer segments win
    SHORT_PIECE_PENALTY = 6.0  # single/double-char "words" are cheap to fake-match;
                                 # tax them so real multi-char words are preferred

    for i in range(1, n + 1):
        for j in range(max(0, i - max_word_len), i):
            if best_score[j] == NEG_INF:
                continue
            piece = run[j:i]
            if piece in wordlist and len(piece) >= 2:
                piece_score = len(piece) * 1.5  # reward longer dictionary hits
            elif piece in wordlist:
                piece_score = -SHORT_PIECE_PENALTY
            else:
                piece_score = bigram_logprob(piece, bigram_model) * len(piece)
            piece_score -= SEGMENT_PENALTY
            cand = best_score[j] + piece_score
            if cand > best_score[i]:
                best_score[i] = cand
                best_split[i] = j

    if best_score[n] == NEG_INF:
        return None

    words, i = [], n
    while i > 0:
        j = best_split[i]
        words.append(run[j:i])
        i = j
    words.reverse()
    return words, best_score[n]


# --------------------------------------------------------------------------- #
# Full pipeline pass
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


def validate_and_repair(text: str, min_len_for_segmentation: int = 4) -> ValidationResult:
    """
    Runs the full word-level gate over `text` (already through your existing
    NFC-normalize + matra-reorder cleaning). Returns repaired text plus a
    flags list — anything in `flags` should go to your OCR-fallback queue,
    not straight to the output JSON.

    Key design choice: every non-dictionary token is run through the SAME
    Viterbi segmenter, whether or not it looks like it might be one word or
    several. The segmenter's own scoring decides — a single-piece result
    means "trust it as one uncatalogued word", 2+ pieces means "this was
    probably a missing space", and no viable split means "flag it". This
    avoids the earlier design's bug: a separate bigram-threshold pre-check
    let merged words with locally-common letter pairs slip through as
    'confident' before the segmenter ever got a chance to look at them.
    """
    wordlist = load_wordlist()
    bigram_model = build_bigram_model()

    out_tokens: list[str] = []
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
            out_tokens.append(raw_token)
            continue

        token_nfc = unicodedata.normalize("NFC", token)

        if is_numeric_or_symbolic(token_nfc):
            out_tokens.append(raw_token)
            continue

        stray = find_stray_chars(token_nfc)
        if stray:
            flags.append(Flag(raw_token, offset, "stray_non_malayalam_char"))
            out_tokens.append(raw_token)  # keep as-is; downstream OCR queue fixes it
            continue

        if token_nfc in wordlist:
            out_tokens.append(raw_token)
            continue

        if len(token_nfc) < min_len_for_segmentation:
            # Too short to segment meaningfully and not a dictionary hit —
            # low confidence either way, flag rather than guess.
            flags.append(Flag(raw_token, offset,
                               reason="unparseable_possible_dropped_glyph"))
            out_tokens.append(raw_token)
            continue

        seg = viterbi_segment(token_nfc, wordlist, bigram_model)
        if seg is None:
            flags.append(Flag(raw_token, offset,
                               reason="unparseable_possible_dropped_glyph"))
            out_tokens.append(raw_token)
            continue

        words, _score = seg
        avg_len = sum(len(w) for w in words) / len(words)

        if len(words) == 1:
            # Segmenter agrees the whole token is best read as one piece —
            # trust it as an uncatalogued inflected/compound word.
            out_tokens.append(prefix + words[0] + suffix)
            continue

        if avg_len >= 2.5:
            # Segmenter found a credible multi-word split.
            out_tokens.append(prefix + " ".join(words) + suffix)
            flags.append(Flag(
                raw_token, offset,
                reason="repaired_missing_space",
                suggestion=" ".join(words),
            ))
            continue

        # Only degenerate (mostly single/double-char) splits were found:
        # neither "it's one word" nor "it's N clean words" holds up.
        # Likely a dropped/garbled glyph broke the word past the point
        # a dictionary+bigram model can safely guess. Flag, don't guess.
        flags.append(Flag(raw_token, offset,
                           reason="unparseable_possible_dropped_glyph"))
        out_tokens.append(raw_token)

    repaired = " ".join(out_tokens)
    return ValidationResult(repaired_text=repaired, flags=flags)


# --------------------------------------------------------------------------- #
# Self-test against the failure patterns from the screenshots
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    samples = [
        "5, 9, 13, 17,... എന്ന സമാന്തര ശ്രേ{ണിയുടെ ആദ്യ 20 പദങ്ങളുടെ തുക എത്ര?",
        "ഇന്ത്യൻ പാർലമെന്റ് 2021 പ്രകാരമുള്ള നിയമങ്ങൾഎത്ര ചാപ്റ്ററുകളായി തിരിച്ചിരിക്കു?",
        "താഴെപറയുന്നവയിൽ വെസലിൽപയോഗിക്കുന്ന ഹാന്റ് ടൂൾ അല്ലാത്തത് ഏത്?",
        "ഒരുഇൻസ്റ്റലേഷന്റെ ഇൻസുലേഷൻറെ സിസ്റ്റൻസ് അളക്കാൻ പൊതുവെ ഉപയോഗിക്കുന്ന ഉപകരണം ഏത്?",
    ]
    for s in samples:
        result = validate_and_repair(s)
        print("INPUT :", s)
        print("OUTPUT:", result.repaired_text)
        for f in result.flags:
            print(f"  FLAG  [{f.reason}] '{f.span}' @ {f.char_offset}"
                  + (f"  -> suggest: '{f.suggestion}'" if f.suggestion else ""))
        print()
