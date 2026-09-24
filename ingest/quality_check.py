"""
Per-question quality scoring, failure diagnosis, and structural anomaly detection.

Replaces the previous page-level sanity checking with fine-grained per-question
analysis that classifies the ROOT CAUSE of each failure before choosing a fix tier.

Cost: $0 — all checks are pure regex/string analysis on already-extracted text.
Vision/LLM calls are NEVER made here; this module only diagnoses.
"""

import re
import os
from dataclasses import dataclass, field
from typing import Literal, Optional
from PIL import Image
import io


# ---------------------------------------------------------------------------
# Quality Score per question
# ---------------------------------------------------------------------------

@dataclass
class QualityScore:
    """Per-question quality assessment result."""
    coverage_pct: float = 1.0
    unmapped_glyph_count: int = 0
    suspicious_gap_count: int = 0
    structural_anomalies: list[str] = field(default_factory=list)
    validation_flags: list = field(default_factory=list)
    status: Literal["ok", "font_issue", "structural_issue", "unreadable", "word_level_corruption"] = "ok"


def score_question_text(raw_text: str, has_malayalam_expected: bool = True, page_flags: list = None) -> QualityScore:
    """
    Runs sanity analysis per-question (not per-page). Reuses the same
    coverage/artifact logic as sanity_check_malayalam_text() but returns
    a structured QualityScore instead of a simple bool.

    Cost: $0 — pure regex and string analysis.
    """
    score = QualityScore()

    if not raw_text or not raw_text.strip():
        return score  # empty is trivially ok

    # Strip HTML tags (inline equation images, block diagrams) before analysis
    text_only = re.sub(r'<[^>]+>', ' ', raw_text)
    # CRITICAL: Mask out LaTeX math expressions ($...$ and $$...$$) before analyzing for unmapped glyphs/wedges
    text_no_math = re.sub(r'\$\$[\s\S]*?\$\$|\$[^$\n]+\$', ' ', text_only)
    non_ws = re.sub(r'\s+', '', text_no_math)

    if not non_ws:
        return score

    malayalam_chars = len(re.findall(r'[\u0D00-\u0D7F]', non_ws))

    # If no Malayalam expected or found, skip Malayalam-specific checks
    if malayalam_chars == 0 and not has_malayalam_expected:
        return score

    if malayalam_chars == 0 and has_malayalam_expected:
        # Expected Malayalam but found none — possible total font failure
        score.status = "font_issue"
        score.coverage_pct = 0.0
        return score

    # 1. Count unmapped glyphs: ASCII/symbols wedged inside Malayalam words (excluding LaTeX formulas)
    wedged = re.findall(
        r'[\u0D00-\u0D7F]'
        r'([A-Za-z#$%&*+<>=@\\^_`|~{}\[\]!/]+)'
        r'[\u0D00-\u0D7F]',
        text_no_math
    )
    score.unmapped_glyph_count = len(wedged)

    # 2. Overall coverage — valid chars should be predominantly Malayalam + basic punctuation
    valid_chars = len(re.findall(
        r'[\u0D00-\u0D7F\dA-Za-z\.,\?!\-\(\)\[\]:;\"\'\u200d<>=/&_ ]', non_ws
    ))
    score.coverage_pct = valid_chars / len(non_ws) if non_ws else 1.0

    # 3. Suspicious gaps: spaces inside what should be a single Malayalam word
    # (consonant + space + combining sign = broken cluster)
    gaps = re.findall(r'[\u0D15-\u0D3A]\s+[\u0D3E-\u0D4E\u0D57]', text_no_math)
    score.suspicious_gap_count = len(gaps)

    # 4. Run structural anomaly detection
    score.structural_anomalies = flag_structural_anomalies(text_only)

    # 4.5 Check if any validation flags from the page apply to this question
    if page_flags:
        for flag in page_flags:
            if flag.span in raw_text or (hasattr(flag, 'suggestion') and flag.suggestion and flag.suggestion in raw_text):
                score.validation_flags.append(flag)

    # 5. Classify overall status
    if score.unmapped_glyph_count > 0:
        score.status = "font_issue"
    elif score.validation_flags:
        score.status = "word_level_corruption"
    elif score.coverage_pct < 0.85:
        score.status = "font_issue"
    elif score.suspicious_gap_count > 3 or len(score.structural_anomalies) > 2:
        score.status = "structural_issue"
    elif score.coverage_pct < 0.92 and score.suspicious_gap_count > 0:
        score.status = "structural_issue"
    else:
        score.status = "ok"

    return score


# ---------------------------------------------------------------------------
# Failure type diagnosis
# ---------------------------------------------------------------------------

FailureType = Literal[
    "unmapped_glyph",
    "reordering_bug",
    "band_merge_bug",
    "equation_crop_fail",
    "unknown_font",
    "scan_quality",
    "word_level_corruption",
]


def diagnose_failure_type(
    question_text: str,
    page_spans: list[dict],
    font_map: dict,
    unmapped_codes: list[dict] = None,
    has_text_layer: bool = True,
    validation_flags: list = None,
) -> str:
    """
    Classifies the root cause of a failing question.

    Returns one of:
      "unmapped_glyph"     — font_mapper returned None/placeholder for specific char codes
      "reordering_bug"     — orphaned matras not adjacent to valid consonant cluster
      "band_merge_bug"     — option markers (A)(B)(C)(D) appear out of sequence
      "equation_crop_fail" — equation image tags are malformed or out of position
      "unknown_font"       — font not in known families and not resolvable
      "scan_quality"       — genuinely unreadable scan (image-only page, low coverage)

    Cost: $0 — pure analysis.
    """
    text = re.sub(r'<[^>]+>', ' ', question_text)
    # Mask out LaTeX math expressions so formulas are never flagged as wedged ASCII
    text = re.sub(r'\$\$[\s\S]*?\$\$|\$[^$\n]+\$', ' ', text)

    # 1. Check for unmapped glyphs (ASCII chars wedged in Malayalam context)
    if unmapped_codes and len(unmapped_codes) > 0:
        return "unmapped_glyph"
        
    if validation_flags and len(validation_flags) > 0:
        return "word_level_corruption"

    wedged = re.findall(
        r'[\u0D00-\u0D7F]\s*'
        r'[A-Za-z#$%&*+<>=@\\^_`|~{}\[\]!/]+'
        r'\s*[\u0D00-\u0D7F]',
        text
    )
    if wedged:
        return "unmapped_glyph"

    # 2. Check for reordering bugs: orphaned pre-base matras
    # Orphaned vowel sign (െ/േ/ൈ) not preceded by a valid consonant within 1 char
    orphaned_matras = re.findall(r'(?:^|[^ക-ഹ\u0D7A-\u0D7F്])([െേൈ])', text)
    if len(orphaned_matras) >= 1:
        return "reordering_bug"

    # 3. Check for band merge bugs: option markers out of order
    option_positions = []
    for m in re.finditer(r'(?:^|\s|\()([A-D])(?:\)|\.|:-)\s', text):
        option_positions.append(m.group(1))
    if len(option_positions) >= 2:
        expected_order = sorted(option_positions, key=lambda x: "ABCD".index(x))
        if option_positions != expected_order:
            return "band_merge_bug"

    # 4. Check for equation crop failures or broken LaTeX
    eq_tags = re.findall(r'<img[^>]*alt="math"[^>]*/>', question_text)
    for tag in eq_tags:
        # CDN-hosted images (R2/Cloudflare) are always valid
        if re.search(r'src="https?://', tag):
            continue
        # Base64 images: check for near-empty data = blank crop
        src_match = re.search(r'src="data:image/png;base64,([^"]+)"', tag)
        if src_match:
            b64_data = src_match.group(1)
            if len(b64_data) < 50:  # near-empty base64 = blank crop
                return "equation_crop_fail"

    # Flag unclosed inline LaTeX math markers
    if question_text.count("$") % 2 != 0:
        return "equation_crop_fail"

    # 5. Check for scan quality issues (image-only, no text layer)
    if not has_text_layer:
        non_ws = re.sub(r'\s+', '', text)
        if not non_ws or len(non_ws) < 10:
            return "scan_quality"

    # 6. If font_map is completely empty for this page, unknown font
    if not font_map:
        malayalam_chars = len(re.findall(r'[\u0D00-\u0D7F]', text))
        if malayalam_chars < 5:
            return "unknown_font"

    # Default: if we got here with a failing question, classify as unmapped_glyph
    return "unmapped_glyph"


# ---------------------------------------------------------------------------
# Structural anomaly detection (zero-cost pattern checks)
# ---------------------------------------------------------------------------

def flag_structural_anomalies(cleaned_text: str) -> list[str]:
    """
    Zero-cost pattern checks for structural bugs in extracted text.

    Returns a list of anomaly tags per question:
      - "orphaned_matra"      — vowel sign with no preceding valid consonant
      - "options_out_of_order" — (A) appears after (B) in char position
      - "repeated_substring"  — repeated char within 3 positions (e.g. "ററററ")
      - "isolated_latin_char" — single Latin char inside Malayalam-majority sentence
                                that isn't a known math variable (x, y, θ)

    Cost: $0 — pure regex.
    """
    anomalies = []

    # Strip HTML tags and mask LaTeX math expressions
    cleaned_text = re.sub(r'<[^>]+>', ' ', cleaned_text)
    cleaned_text = re.sub(r'\$\$[\s\S]*?\$\$|\$[^$\n]+\$', ' ', cleaned_text)

    # 1. Orphaned matra / combining sign: vowel sign (െ/േ/ൈ/ൊ/ോ/ൌ) or anusvara (ം)
    #    with no preceding valid consonant cluster within 1 char, or attached to Latin variable (e.g. 'A' ◌ം)
    orphaned = re.findall(r'(?:^|\s)([െേൈ])', cleaned_text)
    if orphaned:
        anomalies.append("orphaned_matra")
    if re.search(r"['\"]?[A-Za-z0-9]['\"]?[\\s\\u25CC]*[-–—]?[\\s\\u25CC]*ം", cleaned_text):
        anomalies.append("orphaned_anusvara")

    # 2. Option marker out of order: (A) appears after (B) in char position
    markers = []
    for m in re.finditer(r'(?:^|\s|\()([A-D])(?:\)|\.|:-)', cleaned_text):
        markers.append((m.start(), m.group(1)))
    if len(markers) >= 2:
        letters = [m[1] for m in markers]
        expected = sorted(letters, key=lambda x: "ABCD".index(x))
        if letters != expected:
            anomalies.append("options_out_of_order")

    # 3. Repeated Malayalam consonant within 3 chars (likely duplicate span)
    if re.search(r'([\u0D15-\u0D3A])\1{3,}', cleaned_text):
        anomalies.append("repeated_substring")

    # 4. Isolated single Latin char inside Malayalam-majority sentence
    #    that isn't a known math/science variable or unit
    # First mask out LaTeX math spans so formulas inside $...$ are never checked as isolated Latin chars
    text_no_math = re.sub(r'\$\$[\s\S]*?\$\$|\$[^$\n]+\$', ' ', cleaned_text)
    known_math_vars = set("abcdefghijklmnopqrstuvwxyzθαβγδεφηθλμπστω")
    known_science_tokens = {"ph", "ml", "mg", "kg", "km", "cm", "mm", "mol", "dna", "rna", "atp", "co2", "h2o", "nacl"}
    words = text_no_math.split()
    if words:
        total_mal_words = sum(1 for w in words if re.search(r'[\u0D00-\u0D7F]', w))
        if total_mal_words > len(words) * 0.5:
            for w in words:
                # Option markers like (A), (B), A., B. are structural labels, not isolated glyph leaks
                if re.match(r"^\(?[A-D]\)?[\.:-]?$", w):
                    continue
                stripped = w.strip(".,;:!?()[]\"'{}<>/\\")
                if stripped.lower() in known_science_tokens:
                    continue
                if len(stripped) == 1 and stripped.isalpha() and stripped.lower() not in known_math_vars:
                    anomalies.append("isolated_latin_char")
                    break
    return anomalies


# ---------------------------------------------------------------------------
# Equation crop validation
# ---------------------------------------------------------------------------

def validate_equation_crop(crop_png_bytes: bytes, bbox: tuple) -> bool:
    """
    Validates an equation image crop before splicing into text.

    Checks:
    1. Blank-check: grayscale pixel variance below threshold = failed crop
    2. Aspect-ratio sanity: width:height or height:width > 10:1 = detection error
    3. Size sanity: crop smaller than 8x8pt = noise

    Returns True if crop is valid, False if it should be dropped.

    Cost: $0 — pure pixel analysis.
    """
    if not crop_png_bytes or len(crop_png_bytes) < 100:
        return False

    try:
        img = Image.open(io.BytesIO(crop_png_bytes))
        w, h = img.size

        # Size sanity: too small = noise
        if w < 8 or h < 8:
            return False

        # Aspect ratio sanity:
        # Vertical slivers (height > 10x width) are almost always column dividers or margin lines
        if h > 10 * w:
            return False
        # Horizontal thin lines (e.g. page borders, divider lines)
        if w > 35 * h or (h < 15 and w > 15 * h):
            return False

        # Blank check: convert to grayscale, check pixel variance
        gray = img.convert("L")
        try:
            pixels = list(gray.get_flattened_data())
        except AttributeError:
            pixels = list(gray.getdata())
        if not pixels:
            return False

        mean_val = sum(pixels) / len(pixels)
        variance = sum((p - mean_val) ** 2 for p in pixels) / len(pixels)

        # Near-uniform variance = blank/failed crop
        # Threshold: variance < 50 means nearly uniform (blank white or blank black)
        if variance < 50:
            return False

        return True

    except Exception:
        return False
