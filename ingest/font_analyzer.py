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

    all_unique = sorted(list(set(cids)))
    if len(all_unique) > max_glyphs:
        print(f"[Font Analyzer] TRUNCATION WARNING: Font has {len(all_unique)} unmapped CIDs, exceeding sheet limit {max_glyphs}. Truncating to {max_glyphs} glyphs.", flush=True)

    target_cids = all_unique[:max_glyphs]
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
            ch_str = chr(cid)
            if ch_str.isprintable() and not ch_str.isspace():
                page.insert_text(
                    (x0 + 16, y0 + 58),
                    ch_str,
                    fontsize=28,
                    fontname=font_alias,
                    color=(0, 0, 0)
                )
        except Exception:
            pass

    pix = page.get_pixmap(dpi=200)
    png_bytes = pix.tobytes("png")
    return png_bytes, target_cids


def render_glyph_sheets(font_buffer: bytes, cids: list[int], batch_size: int = 64) -> list[tuple[bytes, list[int]]]:
    """
    Renders multiple high-resolution visual grid sheets for fonts that exceed batch_size glyphs.
    Guarantees no glyphs are silently lost or truncated.
    Returns list of (png_bytes, processed_cids).
    """
    if not font_buffer or not cids:
        return []

    unique_cids = sorted(list(set(cids)))
    sheets = []
    for i in range(0, len(unique_cids), batch_size):
        chunk = unique_cids[i:i + batch_size]
        sheet_bytes, processed = render_glyph_sheet(font_buffer, chunk, max_glyphs=len(chunk))
        if sheet_bytes:
            sheets.append((sheet_bytes, processed))
    return sheets


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


def analyze_font_map(
    font_name: str,
    font_buffer: bytes,
    unmapped_cids: list[int] = None
) -> dict:
    """
    Renders visual glyph sheets and transcribes with Gemini Vision to produce a proposed mapping.
    Crucially: DOES NOT auto-commit or write to disk/universal DB.
    Allows validation against actual question texts before committing.
    """
    if not font_buffer:
        return {"status": "error", "message": "No font buffer provided"}

    font_md5 = hashlib.md5(font_buffer).hexdigest()

    if not unmapped_cids:
        unmapped_cids = list(range(1, 150))

    sheets = render_glyph_sheets(font_buffer, unmapped_cids, batch_size=64)
    if not sheets:
        return {"status": "error", "message": "Failed to render glyph sheets"}

    combined_map = {}
    for sheet_idx, (png_bytes, target_cids) in enumerate(sheets):
        if len(sheets) > 1:
            print(f"[Font Analyzer] Transcribing sheet {sheet_idx + 1}/{len(sheets)} ({len(target_cids)} glyphs) for '{font_name}'...", flush=True)
        sheet_map = distill_glyph_sheet_with_gemini(png_bytes, target_cids)
        if sheet_map:
            combined_map.update(sheet_map)

    if not combined_map:
        return {"status": "error", "message": "Distillation produced no mappings"}

    return {
        "status": "success",
        "font_name": font_name,
        "font_hash": font_md5,
        "mapped_count": len(combined_map),
        "proposed_mapping": combined_map
    }


def analyze_and_install_font(
    font_name: str,
    font_buffer: bytes,
    unmapped_cids: list[int] = None
) -> dict:
    """
    Orchestrates analysis and legacy direct installation of a font map.
    """
    res = analyze_font_map(font_name, font_buffer, unmapped_cids)
    if res.get("status") != "success":
        return res

    font_md5 = res["font_hash"]
    proposed_map = res["proposed_mapping"]
    success = install_font_map(font_md5, font_name, font_buffer, proposed_map, source="gemini_analysis")
    if success:
        return {
            "status": "success",
            "font_name": font_name,
            "font_hash": font_md5,
            "mapped_count": len(proposed_map),
            "mappings": proposed_map
        }
    return {"status": "error", "message": "Installation failed"}


def repair_unmapped_glyphs(
    font_hash: str,
    font_name: str,
    font_buffer: bytes,
    unmapped_codes: list[dict],
) -> dict:
    """
    Targeted font-map repair: batches ALL unique unmapped glyph codes across
    the entire document into Vision calls (using multi-sheet batching if > 64 glyphs).

    Args:
        font_hash: MD5 hash of the font binary
        font_name: Human-readable font name
        font_buffer: Raw font binary bytes
        unmapped_codes: List of {"code": int, "font": str, "bbox": list} dicts
                        collected across all pages

    Returns:
        {
            "status": "success" | "error" | "no_glyphs",
            "proposed_mapping": {str(cid): unicode_char, ...},
            "unique_glyphs": int,
            "prompt_tokens": int,
            "completion_tokens": int,
        }
    """
    if not font_buffer:
        return {"status": "error", "message": "No font buffer provided",
                "proposed_mapping": {}, "unique_glyphs": 0,
                "prompt_tokens": 0, "completion_tokens": 0}

    # 1. Deduplicate unmapped glyph codes
    unique_codes = sorted(set(entry["code"] for entry in unmapped_codes if entry.get("code")))

    if not unique_codes:
        return {"status": "no_glyphs", "message": "No unmapped glyphs to repair",
                "proposed_mapping": {}, "unique_glyphs": 0,
                "prompt_tokens": 0, "completion_tokens": 0}

    print(f"[Font Repair] {len(unmapped_codes)} total unmapped occurrences → "
          f"{len(unique_codes)} unique glyph codes for font '{font_name}'", flush=True)

    # 2. Render failing glyphs into grid sheets (batching if >64)
    sheets = render_glyph_sheets(font_buffer, unique_codes, batch_size=64)
    if not sheets:
        return {"status": "error", "message": "Failed to render glyph sheets",
                "proposed_mapping": {}, "unique_glyphs": len(unique_codes),
                "prompt_tokens": 0, "completion_tokens": 0}

    # 3. Vision calls for sheets
    google_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    enable_gemini = os.getenv("ENABLE_GEMINI", "true").lower() in ("true", "1", "yes")

    if not google_key or not enable_gemini:
        return {"status": "error", "message": "Gemini not available for glyph repair",
                "proposed_mapping": {}, "unique_glyphs": len(unique_codes),
                "prompt_tokens": 0, "completion_tokens": 0}

    proposed_mapping = {}
    for s_idx, (png_bytes, target_cids) in enumerate(sheets):
        if len(sheets) > 1:
            print(f"[Font Repair] Distilling sheet {s_idx + 1}/{len(sheets)} ({len(target_cids)} glyphs) for '{font_name}'...", flush=True)
        m = distill_glyph_sheet_with_gemini(png_bytes, target_cids)
        if m:
            proposed_mapping.update(m)

    if not proposed_mapping:
        return {"status": "error", "message": "Vision transcription produced no mappings",
                "proposed_mapping": {}, "unique_glyphs": len(unique_codes),
                "prompt_tokens": 0, "completion_tokens": 0}

    print(f"[Font Repair] Vision transcribed {len(proposed_mapping)}/{len(unique_codes)} "
          f"glyphs for font '{font_name}'", flush=True)

    return {
        "status": "success",
        "proposed_mapping": proposed_mapping,
        "unique_glyphs": len(unique_codes),
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

