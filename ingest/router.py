"""
Universal PDF text extraction router.

Extraction strategy per page:
1. Detect ALL subset fonts → build/retrieve CID→Unicode maps (font_mapper)
2. Rebuild text using font maps (rawdict char-by-char)
3. Clean text (normalize, visual-to-logical reorder)
4. Sanity check → if garbled, trigger full-page Vision OCR fallback
5. Process inline equation images via Mathpix/Gemini equation OCR
"""

import pymupdf
import unicodedata
import os
import base64
import requests
import re
from google import genai
from google.genai import types
import pytesseract
from PIL import Image
import io


_quota_exhausted = False


def clean_fraction_term(s: str) -> str:
    s = s.strip().strip("`'\"-._:")
    subs = {
        'A': '4', 'a': '4',
        'S': '5', 's': '5',
        'O': '0', 'o': '0',
        'i': '1', 'I': '1', 'l': '1', '|': '1', '!': '1', '/': '1',
        'B': '8', 'b': '6',
        'Z': '2', 'z': '2'
    }
    if len(s) == 1 and s in subs:
        return subs[s]
    s = re.sub(r'^[iIl|!/](?=\d)', '1', s)
    s = re.sub(r'^[Tt](?=\d)', '1', s)
    return s


def ocr_math_image(pil_img: Image.Image) -> str:
    """Local offline math OCR parser using Tesseract with fraction & symbol segmentation."""
    gray = pil_img.convert("L")
    w, h = gray.size

    # Check horizontal segments separated by whitespace
    col_sums = [sum(1 for y in range(h) if gray.getpixel((x, y)) < 128) for x in range(w)]

    segments = []
    start = None
    for x, c in enumerate(col_sums):
        if c > 0 and start is None:
            start = x
        elif c == 0 and start is not None:
            if x - start >= 4:
                segments.append((start, x))
            start = None
    if start is not None and (w - start >= 4):
        segments.append((start, w))

    if len(segments) > 1:
        tokens = []
        for x0, x1 in segments:
            crop = gray.crop((max(0, x0 - 2), 0, min(w, x1 + 2), h))
            cw, ch = crop.size
            row_sums = [sum(1 for cx in range(cw) if crop.getpixel((cx, y)) < 128) for y in range(ch)]
            active_rows = [y for y, r in enumerate(row_sums) if r > 0]

            # Check if this segment is an equals sign '=' (two horizontal bars)
            if active_rows:
                y_span = active_rows[-1] - active_rows[0]
                crop_arr = pytesseract.image_to_string(crop, config="--psm 10").strip()
                if "=" in crop_arr or "-" in crop_arr or (y_span < ch * 0.35 and cw > ch * 0.3):
                    tokens.append("=")
                    continue

            raw = pytesseract.image_to_string(crop, config="--psm 6").strip()
            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            if len(lines) == 2:
                n = clean_fraction_term(lines[0])
                d = clean_fraction_term(lines[1])
                tokens.append(f"\\frac{{{n}}}{{{d}}}")
            elif len(lines) == 1:
                t = lines[0].strip()
                if t in ("-", "—", "–"): tokens.append("-")
                elif t in ("x", "X", "*"): tokens.append("\\times")
                else: tokens.append(t)
            else:
                tokens.append(raw)
        res = " ".join(tokens).strip()
        res = re.sub(r'\s*\+\s*', ' + ', res)
        res = re.sub(r'\s*=\s*', ' = ', res)
        return res

    # Single segment
    raw = pytesseract.image_to_string(gray, config="--psm 6").strip()
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    if len(lines) == 2:
        n = clean_fraction_term(lines[0])
        d = clean_fraction_term(lines[1])
        return f"\\frac{{{n}}}{{{d}}}"
    single = " ".join(lines).strip()
    m = re.match(r'^([-+0-9a-zA-Z()]+)\s*/\s*([-+0-9a-zA-Z()]+)$', single)
    if m:
        return f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}"
    return single


def ocr_equation_clip(img_bytes: bytes) -> tuple[str, int, int]:
    """
    Routes an equation / symbol image region:
    1. Mathpix API if configured
    2. Gemini Vision OCR if GOOGLE_API_KEY configured and quota available
    3. High-accuracy local Tesseract OCR fallback with mathematical fraction parsing
    """
    global _quota_exhausted

    # 1. Mathpix API
    app_id = os.getenv("MATHPIX_APP_ID")
    app_key = os.getenv("MATHPIX_APP_KEY")
    if app_id and app_key:
        try:
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            response = requests.post(
                "https://api.mathpix.com/v3/text",
                json={"src": f"data:image/png;base64,{img_b64}", "formats": ["text"]},
                headers={"app_id": app_id, "app_key": app_key, "Content-Type": "application/json"},
                timeout=10
            )
            if response.status_code == 200:
                result_text = response.json().get("text", "").strip()
                if result_text:
                    print(f"[Mathpix OCR] Transcribed: {result_text}", flush=True)
                    return result_text, 0, 0
        except Exception as e:
            print(f"[Mathpix OCR] Failed: {e}. Using fallback OCR...", flush=True)

    # 2. Vision OCR for Mathematical Expressions
    enable_gemini = os.getenv("ENABLE_GEMINI", "true").lower() in ("true", "1", "yes")
    google_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if enable_gemini and google_key and not _quota_exhausted:
        try:
            client = genai.Client(api_key=google_key)
            model_name = os.getenv("GEMINI_EXTRACT_MODEL", "gemini-3.6-flash")
            res = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    "Transcribe the mathematical formula or equation line in this snippet into LaTeX. Output ONLY the line with inline LaTeX ($...$), e.g. $\\frac{1}{3}$ or $\\chi^2$. No explanations."
                ]
            )
            text = res.text.strip()
            if "is:" in text:
                text = text.split("is:")[-1].strip()
            text = text.replace("`", "").strip()
            if text:
                print(f"[Equation OCR] Transcribed via Gemini: {text}", flush=True)
                p_tokens = res.usage_metadata.prompt_token_count if res.usage_metadata else 0
                c_tokens = res.usage_metadata.candidates_token_count if res.usage_metadata else 0
                return text, p_tokens, c_tokens
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                _quota_exhausted = True
                print(f"[Equation OCR] Gemini API quota reached. Falling back to local OCR.", flush=True)
            else:
                print(f"[Equation OCR] Gemini OCR failed: {e}. Falling back to local OCR.", flush=True)

    # 3. Local Tesseract OCR fallback
    try:
        pil_img = Image.open(io.BytesIO(img_bytes))
        math = ocr_math_image(pil_img)
        if math:
            print(f"[Equation OCR] Local OCR transcribed: {math}", flush=True)
            return math, 0, 0
    except Exception as e:
        print(f"[Equation OCR] Local OCR failed: {e}", flush=True)

    return "", 0, 0


def ocr_page_vision(img_bytes: bytes) -> tuple[str, int, int]:
    """Uses Gemini Vision API as a fallback to OCR severely garbled pages."""
    global _quota_exhausted
    enable_gemini = os.getenv("ENABLE_GEMINI", "true").lower() in ("true", "1", "yes")
    google_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not enable_gemini or not google_key or _quota_exhausted:
        return "", 0, 0

    try:
        client = genai.Client(api_key=google_key)
        res = client.models.generate_content(
            model=os.getenv("GEMINI_EXTRACT_MODEL", "gemini-3.6-flash"),
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                "Extract all text from this page exactly as it appears. Preserve the Malayalam script, numbers, and layout. Do not add conversational text."
            ]
        )
        text = res.text.strip()
        print(f"[Page Vision OCR] Fallback OCR completed successfully.", flush=True)
        
        p_tokens = res.usage_metadata.prompt_token_count if res.usage_metadata else 0
        c_tokens = res.usage_metadata.candidates_token_count if res.usage_metadata else 0
        return text, p_tokens, c_tokens
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            _quota_exhausted = True
            print(f"[Page Vision OCR] Gemini API quota reached. Disabling further vision OCR calls.", flush=True)
        else:
            print(f"[Page Vision OCR] Failed: {e}", flush=True)
            
    return "", 0, 0


def _rebuild_text_from_font_map(page, font_map: dict, equation_items: list[dict] = None) -> str:
    """
    Rebuilds page text non-destructively using 2D spatial reading order:
    - Preserves all already-valid Malayalam Unicode characters.
    - Preserves natural spacing and Latin text (headers, English terms).
    - Substitutes unmapped ligatures/symbols from the resolved Indic font map.
    - Spatially injects inline math/equation images based on bounding box alignment.
    - Sorts visual line segments top-to-bottom and left-to-right, ensuring that questions
      with separate content streams (English labels in one font stream, Malayalam text in another)
      are correctly assembled on the same line.
    """
    rawdict = page.get_text("rawdict")
    eq_items = list(equation_items) if equation_items else []
    extracted_lines = []

    for b in rawdict.get("blocks", []):
        if "lines" not in b:
            continue
        for l in b["lines"]:
            line_chars = []
            prev_x1 = None
            font_size = 10
            for s in l["spans"]:
                span_font = s["font"]
                base_font = span_font.split("+")[-1]
                font_size = s["size"]
                f_map = font_map.get(span_font) or font_map.get(base_font) if font_map else None

                for c in s["chars"]:
                    ch = c["c"]
                    code = ord(ch) if len(ch) == 1 else 0
                    bbox = c.get("bbox")

                    # Whitespace characters and synthetic spaces must NEVER be mapped through font maps
                    if ch in (" ", "\n", "\t", "\xa0") or c.get("synthetic"):
                        resolved = ch
                    elif f_map and (str(code) in f_map or code in f_map):
                        resolved = f_map.get(str(code)) or f_map.get(code)
                    else:
                        resolved = ch

                    # Combining signs that should never be preceded by a detached space
                    COMBINING_SIGNS = (0x0D02, 0x0D03, 0x0D4D, 0x0D57)
                    is_combining = (
                        bool(resolved) and (
                            (0x0D3E <= ord(resolved[0]) <= 0x0D4E) or 
                            (ord(resolved[0]) in COMBINING_SIGNS) or
                            resolved.startswith("\u0D4D")
                        )
                    )

                    # Natural horizontal spacing gap - only between independent words/letters, never before combining signs!
                    if bbox and prev_x1 is not None:
                        dist = bbox[0] - prev_x1
                        if dist > font_size * 0.18 and not "".join(line_chars).endswith(" "):
                            if not is_combining and resolved not in (" ", "\n", "\xa0"):
                                line_chars.append(" ")

                    line_chars.append(resolved)

                    if bbox:
                        prev_x1 = bbox[2]

            line_str = "".join(line_chars).strip()
            l_bbox = l.get("bbox", [0, 0, 0, 0])
            l_y_center = (l_bbox[1] + l_bbox[3]) / 2

            # Spatial placement for equations aligning with this line
            if eq_items:
                matching = [
                    item for item in eq_items
                    if abs(l_bbox[3] - item["rect"].y1) <= 12 or abs(item["y_center"] - l_y_center) <= 20
                ]
                if matching:
                    matching.sort(key=lambda item: item["x_center"])
                    for m in matching:
                        if m["rect"].x0 >= l_bbox[0] - 10:
                            line_str = f"{line_str} {m['math']}".strip()
                        else:
                            line_str = f"{m['math']} {line_str}".strip()
                        if m in eq_items:
                            eq_items.remove(m)

            if line_str:
                extracted_lines.append({
                    "text": line_str,
                    "bbox": l_bbox,
                    "y_center": l_y_center,
                    "x0": l_bbox[0],
                    "x1": l_bbox[2],
                    "y0": l_bbox[1],
                    "y1": l_bbox[3]
                })

    # Append any remaining unplaced equations
    if eq_items:
        for m in eq_items:
            extracted_lines.append({
                "text": m["math"],
                "bbox": [m["rect"].x0, m["rect"].y0, m["rect"].x1, m["rect"].y1],
                "y_center": m["y_center"],
                "x0": m["rect"].x0,
                "x1": m["rect"].x1,
                "y0": m["rect"].y0,
                "y1": m["rect"].y1
            })

    if not extracted_lines:
        return ""

    # Sort lines vertically primarily
    extracted_lines.sort(key=lambda item: item["y_center"])

    # Group lines that lie within the same visual horizontal band (threshold ~5.5pt)
    bands = []
    for line in extracted_lines:
        placed = False
        for band in bands:
            if abs(line["y_center"] - band["avg_y"]) <= 5.5:
                band["lines"].append(line)
                band["avg_y"] = sum(l["y_center"] for l in band["lines"]) / len(band["lines"])
                placed = True
                break
        if not placed:
            bands.append({
                "avg_y": line["y_center"],
                "lines": [line]
            })

    # Sort bands top-to-bottom
    bands.sort(key=lambda b: b["avg_y"])

    # Within each band, sort lines left-to-right (x0)
    # If the gap between text elements on the same band is large, insert a tab to preserve columns
    page_lines = []
    for band in bands:
        band["lines"].sort(key=lambda l: l["x0"])
        
        band_str = ""
        prev_x1 = None
        
        for l in band["lines"]:
            text = l["text"]
            if not text:
                continue
                
            if prev_x1 is not None:
                gap = l["x0"] - prev_x1
                if gap > 30:  # Significant gap suggests columns or table cells
                    band_str += " \t "
                else:
                    band_str += " "
            
            band_str += text
            prev_x1 = max(prev_x1 if prev_x1 is not None else 0, l["x1"])
            
        if band_str.strip():
            page_lines.append(band_str.strip())

    return "\n".join(page_lines)


def extract_text_from_pdf(file_path: str) -> tuple[str, int, int]:
    """
    Universal PDF text extraction pipeline.
    
    Per page:
    1. Detect inline math/equation images and extract via OCR
    2. Detect subset fonts → build/retrieve font maps
    3. Rebuild text using font maps with inline math injection
    4. Clean and normalize text
    5. Sanity check → if still garbled, trigger full-page Vision OCR fallback
    """
    doc = pymupdf.open(file_path)
    full_pages = []
    total_p_tokens = 0
    total_c_tokens = 0

    for page_idx, page in enumerate(doc):
        images = page.get_images()

        # ---------------------------------------------------------------
        # Step 1: Pre-extract and OCR equation / symbol clips on the page
        # ---------------------------------------------------------------
        equation_items = []
        seen_rect_keys = set()
        if images:
            for img_info in images:
                xref = img_info[0]
                rects = page.get_image_rects(xref)
                if not rects:
                    continue
                for r in rects:
                    r_key = (round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1))
                    if r_key in seen_rect_keys:
                        continue
                    seen_rect_keys.add(r_key)

                    # Skip full-page background images
                    if r.width > page.rect.width * 0.8 and r.height > page.rect.height * 0.5:
                        continue
                    # Skip tiny speckles
                    if r.width < 4 or r.height < 4:
                        continue
                    strip_pix = page.get_pixmap(clip=r, dpi=300)
                    latex, p_tok, c_tok = ocr_equation_clip(strip_pix.tobytes("png"))
                    total_p_tokens += p_tok
                    total_c_tokens += c_tok
                    if latex:
                        clean_math = latex.strip()
                        if not clean_math.startswith("$"):
                            clean_math = f"${clean_math}$"
                        equation_items.append({
                            "rect": r,
                            "math": clean_math,
                            "y_center": (r.y0 + r.y1) / 2,
                            "x_center": (r.x0 + r.x1) / 2
                        })

        # ---------------------------------------------------------------
        # Step 2: Universal font map detection and text rebuild
        # ---------------------------------------------------------------
        from .font_mapper import get_or_create_font_map
        font_map = get_or_create_font_map(doc, page_idx)

        if font_map:
            font_names = list(set(font_map.keys()))
            print(f"Page {page_idx + 1}: {len(font_names)} mapped fonts. Rebuilding text from CIDs...", flush=True)
            page_text = _rebuild_text_from_font_map(page, font_map, equation_items)
        else:
            page_text = _rebuild_text_from_font_map(page, {}, equation_items)

        # ---------------------------------------------------------------
        # Step 3: Clean and normalize
        # ---------------------------------------------------------------
        from .text_cleaner import clean_indic_text, sanity_check_malayalam_text
        page_text = clean_indic_text(page_text)

        # ---------------------------------------------------------------
        # Step 4: FML/ML-TT legacy font heuristic
        # ---------------------------------------------------------------
        malayalam_chars_count = len(re.findall(r'[\u0D00-\u0D7F]', page_text))
        if malayalam_chars_count < 20 and re.search(r'[ß∂Ø∏ƒ≥‰≠‡∑›™‚]', page_text):
            print(f"Page {page_idx + 1}: FML/ML-TT legacy symbols detected. Transliterating...", flush=True)
            from .smc_fml_converter import convert_fml_to_unicode
            page_text = convert_fml_to_unicode(page_text)
            page_text = clean_indic_text(page_text)

        # ---------------------------------------------------------------
        # Step 5: Sanity check + full-page Vision OCR fallback
        # ---------------------------------------------------------------
        is_sane, metrics = sanity_check_malayalam_text(page_text)

        if not is_sane:
            reason = metrics.get('reason', 'unknown')
            match_str = metrics.get('match', '')
            coverage = metrics.get('coverage', 0)
            print(f"Page {page_idx + 1}: Sanity check FAILED "
                  f"(Coverage: {coverage:.1%}, Reason: {reason} {match_str}). "
                  f"Triggering Vision OCR fallback...", flush=True)

            page_pix = page.get_pixmap(dpi=200)
            fallback_text, p_tok, c_tok = ocr_page_vision(page_pix.tobytes("png"))
            total_p_tokens += p_tok
            total_c_tokens += c_tok

            if fallback_text:
                full_pages.append(fallback_text)
                continue

        # If page has no text but has images, attempt fallback
        if not page_text.strip() and images:
            print(f"Page {page_idx + 1}: No text, {len(images)} images. Rendering for OCR...", flush=True)
            page_pix = page.get_pixmap(dpi=200)
            fallback_text, p_tok, c_tok = ocr_page_vision(page_pix.tobytes("png"))
            total_p_tokens += p_tok
            total_c_tokens += c_tok
            if fallback_text:
                full_pages.append(fallback_text)
                continue

        full_pages.append(page_text)

    return "\n\n".join(full_pages), total_p_tokens, total_c_tokens


def route_and_extract(file_path: str) -> tuple[str, int, int]:
    """Ingestion routing layer entrypoint."""
    return extract_text_from_pdf(file_path)
