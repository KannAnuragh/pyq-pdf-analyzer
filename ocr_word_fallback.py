"""
ocr_word_fallback.py

Resolves a FlaggedWord by cropping just its bbox out of the page and running
OCR on that crop — not the whole page. This is the piece that makes the
"offline and lightweight" goal actually hold: instead of every ambiguous
page triggering a full-page Gemini Vision call (your current Case 8 path,
which is neither offline nor free), only the specific words the validator
couldn't parse get sent anywhere, and the offline engine gets first crack.

[Guessing] I don't have your actual PDF-handle plumbing, so `pdf_path` /
`page_number` / `bbox` are assumed to be enough to re-open the page with
PyMuPDF — adjust to however router.py already holds page references during
ingestion (you may already have the fitz.Page object in scope at the point
these flags are generated, in which case skip the reopen and pass it in
directly).

Requires: pip install pytesseract --break-system-packages
          + system tesseract-ocr with the Malayalam (`mal`) trained data
            (apt install tesseract-ocr-mal on Debian/Ubuntu — this is the
            offline, lightweight option; no GPU, no API key).

IMPORTANT — validate this on your own crops before trusting it. Your team
already found whole-page/paragraph Tesseract OCR mangles Malayalam ligatures
(that's why you moved to the contour-hash pipeline in the first place). A
single isolated word is a much easier case for Tesseract than full-paragraph
reading-order reconstruction, but "easier" isn't "solved" — treat the
confidence threshold below as a starting point to tune against real
low-confidence words from your own PDFs, not a number to trust blindly.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None


CROP_DPI = 400            # higher than your page-level 200 DPI (Case 8) —
                           # cheap to go high-res when cropping one word
CROP_PADDING_PT = 4.0     # small margin around the bbox so glyphs aren't clipped
MIN_CONFIDENCE = 60.0     # tesseract's own per-word confidence (0-100);
                           # below this, don't trust the OCR guess at all


@dataclass
class WordOcrResult:
    text: str | None
    confidence: float
    engine: str  # "tesseract" | "none"


def crop_word_image(pdf_path: str | Path, page_number: int, bbox: list[float]) -> Image.Image:
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number]
        x0, y0, x1, y1 = bbox
        rect = fitz.Rect(
            x0 - CROP_PADDING_PT, y0 - CROP_PADDING_PT,
            x1 + CROP_PADDING_PT, y1 + CROP_PADDING_PT,
        )
        zoom = CROP_DPI / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=rect)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return img
    finally:
        doc.close()


def ocr_word_offline(image: Image.Image) -> WordOcrResult:
    if pytesseract is None:
        return WordOcrResult(None, 0.0, "none")

    data = pytesseract.image_to_data(
        image, lang="mal",
        config="--psm 7",  # treat the crop as a single text line
        output_type=pytesseract.Output.DICT,
    )
    words = [w for w in data["text"] if w.strip()]
    if not words:
        return WordOcrResult(None, 0.0, "tesseract")

    confidences = [c for c, w in zip(data["conf"], data["text"]) if w.strip()]
    avg_conf = sum(float(c) for c in confidences) / len(confidences) if confidences else 0.0
    text = "".join(words)  # Malayalam words rarely tokenize as multiple "words" internally
    return WordOcrResult(text, avg_conf, "tesseract")


def resolve_flagged_word(pdf_path: str | Path, page_number: int, bbox: list[float]) -> WordOcrResult:
    """
    Primary entry point. Crops, runs offline OCR, and returns a result you
    can attach to the FlaggedWord row (ocr_candidate, ocr_confidence).

    If confidence is below MIN_CONFIDENCE, don't auto-apply it — surface it
    in the review UI alongside the crop image itself so a human (or your
    existing Gemini-Vision-as-judge path, scoped to this one crop instead
    of the whole page) makes the final call. That's a much cheaper Vision
    call than your current whole-page fallback, so wiring your existing
    Gemini path to accept a cropped image here is a reasonable second tier
    — just don't skip straight to it; let the offline pass filter first.
    """
    image = crop_word_image(pdf_path, page_number, bbox)
    result = ocr_word_offline(image)
    return result
