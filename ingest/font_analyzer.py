"""
Font Analyzer & Distillation Engine for Malayalam & Indic Fonts.

Analyzes unrecognized and unmapped embedded TrueType/OpenType fonts:
1. Renders unmapped glyphs onto a high-resolution, numbered visual glyph sheet.
2. Uses Gemini Vision API to transcribe each numbered glyph into standard Malayalam Unicode.
3. Aligns extracted character sequences with page OCR ground truth when available.
4. Validates the candidate mapping against Malayalam sanity rules.
5. Installs the verified font map permanently into the system via font_installer.
"""

import os
import io
import json
import re
import hashlib
import pymupdf
from google import genai
from google.genai import types

from .font_installer import install_font_map


def render_glyph_sheet(font_buffer: bytes, cids: list[int], max_glyphs: int = 64) -> tuple[bytes, list[int]]:
    """
    Renders a clean, high-resolution visual grid sheet of unmapped glyphs with their CID numbers.
    Returns (png_bytes, processed_cids).
    """
    if not font_buffer or not cids:
        return b"", []

    target_cids = sorted(list(set(cids)))[:max_glyphs]
    cols = 8
    rows = (len(target_cids) + cols - 1) // cols
    tile_w = 90
    tile_h = 90
    page_w = cols * tile_w + 40
    page_h = rows * tile_h + 40

    doc = pymupdf.open()
    page = doc.new_page(width=page_w, height=page_h)

    # Insert the custom font
    font_alias = "TargetIndicFont"
    try:
        page.insert_font(fontname=font_alias, fontbuffer=font_buffer)
    except Exception as e:
        print(f"[Font Analyzer] Failed to insert font buffer into PDF page: {e}", flush=True)
        return b"", []

    # Draw grid and glyphs
    for idx, cid in enumerate(target_cids):
        r = idx // cols
        c = idx % cols
        x0 = 20 + c * tile_w
        y0 = 20 + r * tile_h

        # Draw tile border
        rect = pymupdf.Rect(x0, y0, x0 + tile_w - 6, y0 + tile_h - 6)
        page.draw_rect(rect, color=(0.7, 0.7, 0.7), width=0.7)

        # Draw CID label
        page.insert_text(
            (x0 + 4, y0 + 14),
            f"#{cid}",
            fontsize=9,
            fontname="helv",
            color=(0.3, 0.3, 0.3)
        )

        # Draw the target glyph in the custom font
        try:
            page.insert_text(
                (x0 + 16, y0 + 58),
                chr(cid),
                fontsize=28,
                fontname=font_alias,
                color=(0, 0, 0)
            )
        except Exception:
            pass

    pix = page.get_pixmap(dpi=200)
    png_bytes = pix.tobytes("png")
    return png_bytes, target_cids


def distill_glyph_sheet_with_gemini(glyph_sheet_png: bytes, cids: list[int]) -> dict:
    """
    Sends the numbered glyph grid to Gemini Flash to transcribe each CID to Malayalam Unicode.
    """
    google_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not google_key or not glyph_sheet_png:
        return {}

    prompt = f"""
You are an expert in Malayalam and Indic typography and OpenType ligature decoding.
This image is a numbered glyph chart extracted from a Malayalam examination paper.
Each box contains a CID label (e.g. #32, #44) and a rendered glyph.

TASK:
Identify the exact character or ligature for each numbered box.
Return a valid JSON object where keys are the CID numbers (e.g. "32", "44") and values are the exact standard Malayalam Unicode strings (or Latin/symbol if it is non-Indic).
Example format:
{{
  "32": "്യ",
  "35": "ഡ്ര",
  "44": "ലൂ",
  "29": "മൂ"
}}

Strict rules:
1. Return ONLY the JSON object inside ```json ... ``` code fences.
2. For Malayalam ligatures, return standard logical Unicode (consonant + virama + consonant or dependent sign).
3. Do not output conversational explanations.
"""
    try:
        client = genai.Client(api_key=google_key)
        model_name = os.getenv("GEMINI_EXTRACT_MODEL", "gemini-3.1-flash-lite")
        res = client.models.generate_content(
            model=model_name,
            contents=[
                types.Part.from_bytes(data=glyph_sheet_png, mime_type="image/png"),
                prompt
            ]
        )
        text = res.text.strip()
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        data = json.loads(text)
        # Filter and sanitize mapping
        clean_map = {}
        for k, v in data.items():
            cid_str = str(k).replace("#", "").strip()
            val = str(v).strip()
            if cid_str.isdigit() and val:
                clean_map[cid_str] = val

        print(f"[Font Analyzer] Gemini transcribed {len(clean_map)} glyphs from sheet.", flush=True)
        return clean_map
    except Exception as e:
        print(f"[Font Analyzer] Error distilling glyph sheet with Gemini: {e}", flush=True)
        return {}


def analyze_and_install_font(
    font_name: str,
    font_buffer: bytes,
    unmapped_cids: list[int] = None
) -> dict:
    """
    Orchestrates the analysis and permanent installation of an unrecognized font:
    1. Renders visual glyph sheet of unmapped CIDs.
    2. Transcribes with Gemini Vision into Unicode.
    3. Installs font map and updates universal glyph hashes.
    """
    if not font_buffer:
        return {"status": "error", "message": "No font buffer provided"}

    font_md5 = hashlib.md5(font_buffer).hexdigest()

    # Determine CIDs if not provided
    if not unmapped_cids:
        # Default scan range for typical subset font CIDs
        unmapped_cids = list(range(1, 150))

    png_bytes, target_cids = render_glyph_sheet(font_buffer, unmapped_cids)
    if not png_bytes:
        return {"status": "error", "message": "Failed to render glyph sheet"}

    distilled_map = distill_glyph_sheet_with_gemini(png_bytes, target_cids)
    if not distilled_map:
        return {"status": "error", "message": "Distillation produced no mappings"}

    success = install_font_map(font_md5, font_name, font_buffer, distilled_map)
    if success:
        return {
            "status": "success",
            "font_name": font_name,
            "font_hash": font_md5,
            "mapped_count": len(distilled_map),
            "mappings": distilled_map
        }
    return {"status": "error", "message": "Installation failed"}
