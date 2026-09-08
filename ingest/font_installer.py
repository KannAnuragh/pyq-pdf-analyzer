"""
Font Installer for Malayalam & Indic Fonts.

Handles persistent installation of learned and uploaded fonts:
- Caches CID-to-Unicode maps in storage/font_maps/<hash>_map.json
- Extracts glyph contour hashes and permanently appends to universal_glyph_hashes.json
- Updates the in-memory master hash registry in font_mapper
- Saves font binaries to storage/reference_fonts/
- Updates dynamic font family list in storage/known_indic_families.json
"""

import os
import json
import hashlib
import tempfile
import pymupdf
from fontTools.ttLib import TTFont

STORAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "storage")
CACHE_DIR = os.path.join(STORAGE_DIR, "font_maps")
UNIVERSAL_HASH_MAP = os.path.join(STORAGE_DIR, "universal_glyph_hashes.json")
REFERENCE_FONTS_DIR = os.path.join(STORAGE_DIR, "reference_fonts")
KNOWN_FAMILIES_FILE = os.path.join(STORAGE_DIR, "known_indic_families.json")

DEFAULT_INDIC_FAMILIES = [
    "Rachana", "Meera", "Manjari", "Gayathri", "Dyuthi", "Chilanka",
    "Anjali", "Karumbi", "Keraleeyam", "Suruma", "Uroob", "Raghu",
    "ML-TT", "Karthika", "Revathi", "Ambili", "Panchari", "Haritha",
    "Indulekha", "Nandini", "Uma", "Charaka", "Manorama",
    "Shree", "SHREE-MAL", "Shree-Mal", "Shree-Lipi"
]


def ensure_storage_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(REFERENCE_FONTS_DIR, exist_ok=True)


def get_known_font_families() -> list[str]:
    """Returns all known Indic font family keywords."""
    ensure_storage_dirs()
    if os.path.exists(KNOWN_FAMILIES_FILE):
        try:
            with open(KNOWN_FAMILIES_FILE, "r", encoding="utf-8") as f:
                families = json.load(f)
                return list(set(DEFAULT_INDIC_FAMILIES + families))
        except Exception:
            pass
    return list(DEFAULT_INDIC_FAMILIES)


def register_font_family(family_name: str):
    """Registers a new font family name persistently."""
    if not family_name or len(family_name) < 2:
        return
    families = get_known_font_families()
    clean = family_name.strip()
    if clean not in families:
        families.append(clean)
        try:
            with open(KNOWN_FAMILIES_FILE, "w", encoding="utf-8") as f:
                json.dump(sorted(families), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Font Installer] Failed to save known font families: {e}", flush=True)


def hash_glyph_coordinates(glyph, glyf_table):
    """Normalized MD5 hash of glyph outline contour coordinates."""
    if glyph.isComposite():
        coords, endPts, flags = glyph.getCoordinates(glyf_table)
    else:
        if not hasattr(glyph, 'coordinates') or not glyph.coordinates:
            return None
        coords = glyph.coordinates

    if not coords or len(coords) < 3:
        return None

    xMin = min(pt[0] for pt in coords)
    yMin = min(pt[1] for pt in coords)
    normalized = []
    for (x, y) in coords:
        normalized.extend([x - xMin, y - yMin])
    return hashlib.md5(str(normalized).encode()).hexdigest()


def install_font_map(font_md5: str, font_name: str, font_buffer: bytes, resolved_map: dict) -> bool:
    """
    Installs a newly analyzed or distilled font map:
    1. Writes storage/font_maps/<font_md5>_map.json
    2. Extracts contour hashes for all resolved glyphs and updates universal_glyph_hashes.json
    3. Saves reference TTF
    4. Updates known families
    """
    if not resolved_map:
        return False

    ensure_storage_dirs()
    clean_name = font_name.split("+")[-1] if "+" in font_name else font_name

    # 1. Save font map cache
    cache_file = os.path.join(CACHE_DIR, f"{font_md5}_map.json")
    try:
        existing_map = {}
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                existing_map = json.load(f)
        existing_map.update({str(k): v for k, v in resolved_map.items()})
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(existing_map, f, ensure_ascii=False, indent=2)
        print(f"[Font Installer] Saved {len(existing_map)} mapped glyphs to {cache_file}", flush=True)
    except Exception as e:
        print(f"[Font Installer] Error saving font map: {e}", flush=True)
        return False

    # 2. Extract contour hashes and update universal glyph database
    if font_buffer:
        try:
            with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as tmp:
                tmp.write(font_buffer)
                tmp_path = tmp.name

            tt = TTFont(tmp_path)
            if "glyf" in tt:
                glyf = tt["glyf"]
                glyph_order = tt.getGlyphOrder()
                new_hashes = {}

                for code_str, unicode_char in resolved_map.items():
                    try:
                        code = int(code_str)
                    except ValueError:
                        continue
                    if code < len(glyph_order):
                        gname = glyph_order[code]
                        if gname in glyf:
                            g_hash = hash_glyph_coordinates(glyf[gname], glyf)
                            if g_hash:
                                new_hashes[g_hash] = unicode_char

                if new_hashes:
                    _append_to_universal_database(new_hashes)

            os.remove(tmp_path)
        except Exception as e:
            print(f"[Font Installer] Error updating glyph hashes: {e}", flush=True)

        # 3. Save font binary
        ref_filename = f"{clean_name.replace(' ', '_')}_{font_md5[:8]}.ttf"
        ref_path = os.path.join(REFERENCE_FONTS_DIR, ref_filename)
        if not os.path.exists(ref_path):
            try:
                with open(ref_path, "wb") as rf:
                    rf.write(font_buffer)
                print(f"[Font Installer] Preserved reference font binary: {ref_path}", flush=True)
            except Exception as e:
                print(f"[Font Installer] Failed to save reference font binary: {e}", flush=True)

    # 4. Register family
    family_base = clean_name.split("-")[0]
    register_font_family(family_base)

    # 5. Invalidate in-memory font_mapper cache so subsequent pages immediately use the newly learned font
    from . import font_mapper
    font_mapper._MASTER_HASHES = None
    font_mapper._TOPO_SIGNATURES = None

    return True


def _append_to_universal_database(new_hashes: dict):
    """Appends newly learned glyph hashes to universal_glyph_hashes.json."""
    data = {"glyph_hashes": {}, "topological_signatures": {}}
    if os.path.exists(UNIVERSAL_HASH_MAP):
        try:
            with open(UNIVERSAL_HASH_MAP, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass

    existing_hashes = data.get("glyph_hashes", {})
    added_count = 0
    for h, u in new_hashes.items():
        if h not in existing_hashes:
            existing_hashes[h] = u
            added_count += 1

    data["glyph_hashes"] = existing_hashes
    try:
        with open(UNIVERSAL_HASH_MAP, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[Font Installer] Added {added_count} new glyph hashes to Universal Master Database.", flush=True)
    except Exception as e:
        print(f"[Font Installer] Failed to update universal glyph database: {e}", flush=True)


def install_reference_font(file_bytes: bytes, filename: str) -> dict:
    """
    Decompiles and permanently installs an uploaded TTF or OTF font file into the system:
    - Extracts its /cmap and GSUB ligature definitions
    - Generates contour hashes for all glyphs
    - Appends them to universal_glyph_hashes.json
    - Saves the font binary to storage/reference_fonts/
    """
    ensure_storage_dirs()
    clean_base = os.path.splitext(filename)[0]

    with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        tt = TTFont(tmp_path)
        if "glyf" not in tt:
            return {"status": "error", "message": "Font does not contain a TrueType 'glyf' table."}

        from .font_mapper import build_glyph_to_unicode_map
        g2u = build_glyph_to_unicode_map(tt)
        glyf = tt["glyf"]

        new_hashes = {}
        for gname in tt.getGlyphOrder():
            if gname in glyf and gname in g2u:
                ghash = hash_glyph_coordinates(glyf[gname], glyf)
                if ghash:
                    new_hashes[ghash] = g2u[gname]

        if new_hashes:
            _append_to_universal_database(new_hashes)

        # Save to reference fonts
        dest_path = os.path.join(REFERENCE_FONTS_DIR, filename)
        with open(dest_path, "wb") as f:
            f.write(file_bytes)

        # Register family
        fam = clean_base.split("-")[0]
        register_font_family(fam)

        # Invalidate in-memory font_mapper cache
        from . import font_mapper
        font_mapper._MASTER_HASHES = None
        font_mapper._TOPO_SIGNATURES = None

        return {
            "status": "success",
            "font_name": clean_base,
            "glyphs_indexed": len(new_hashes),
            "reference_file": dest_path
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def list_installed_fonts() -> list[dict]:
    """Lists all installed font maps, reference fonts, and system stats."""
    ensure_storage_dirs()
    results = []

    # Read font maps
    if os.path.exists(CACHE_DIR):
        for fname in os.listdir(CACHE_DIR):
            if fname.endswith("_map.json"):
                fpath = os.path.join(CACHE_DIR, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        m = json.load(f)
                    font_hash = fname.replace("_map.json", "")
                    results.append({
                        "type": "font_map",
                        "font_hash": font_hash,
                        "file_name": fname,
                        "mapped_glyphs": len(m)
                    })
                except Exception:
                    pass

    # Read reference fonts
    if os.path.exists(REFERENCE_FONTS_DIR):
        for fname in os.listdir(REFERENCE_FONTS_DIR):
            if fname.lower().endswith((".ttf", ".otf")):
                fpath = os.path.join(REFERENCE_FONTS_DIR, fname)
                results.append({
                    "type": "reference_font",
                    "file_name": fname,
                    "size_bytes": os.path.getsize(fpath)
                })

    return results
