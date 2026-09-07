"""
Universal Font Mapper for Malayalam PDF extraction.

Handles ANY subset/embedded Malayalam font by:
1. Shape-matching against the comprehensive Universal Glyph Hash database (10,800+ glyphs
   covering Rachana, Meera, Manjari, Gayathri, Chilanka, Dyuthi, AnjaliOldLipi, Keraleeyam,
   Karumbi, Suruma, Uroob, Noto Sans/Serif, ML-TT Karthika).
2. Topological signature matching for font version variations.
3. Preserving PDF /ToUnicode CMaps so valid Unicode is never stripped or corrupted.
4. Permanent caching of resolved font maps by font data MD5 hash.
"""

import pymupdf
from fontTools.ttLib import TTFont
import hashlib
import os
import json
import tempfile

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "storage", "font_maps")
UNIVERSAL_HASH_MAP = os.path.join(os.path.dirname(__file__), "..", "storage", "universal_glyph_hashes.json")
RACHANA_HASH_MAP = os.path.join(os.path.dirname(__file__), "rachana_hash_map.json")
MEERA_HASH_MAP = os.path.join(os.path.dirname(__file__), "meera_hash_map.json")
REFERENCE_FONTS_DIR = os.path.join(os.path.dirname(__file__), "..", "storage", "reference_fonts")
RACHANA_FONT_PATH = os.path.join(os.path.dirname(__file__), "..", "rachana", "ttf", "Rachana-Regular.ttf")

INDIC_FAMILIES = (
    "Rachana", "Meera", "Manjari", "Gayathri", "Dyuthi", "Chilanka",
    "Anjali", "Karumbi", "Keraleeyam", "Suruma", "Uroob", "Raghu",
    "ML-TT", "Karthika", "Revathi", "Ambili", "Panchari", "Haritha",
    "Indulekha", "Nandini", "Uma", "Charaka", "Manorama"
)

_MASTER_HASHES = None
_TOPO_SIGNATURES = None


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def hash_font_data(buffer: bytes) -> str:
    """Hash the raw font binary to create a stable, unique cache key."""
    return hashlib.md5(buffer).hexdigest()


def hash_glyph_coordinates(glyph, glyf_table):
    """Creates a unique hash for a glyph based on its normalized contour coordinates."""
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


def get_glyph_topology_signature(glyph, glyf_table):
    """
    Returns a robust geometric topology signature (contour_count, point_count, endPts, aspect_ratio)
    that is resilient to slight coordinate nudges between font versions.
    """
    if glyph.isComposite():
        coords, endPts, flags = glyph.getCoordinates(glyf_table)
        num_contours = len(endPts) if endPts else 0
    else:
        if not hasattr(glyph, 'coordinates') or not glyph.coordinates:
            return None
        coords = glyph.coordinates
        endPts = glyph.endPtsOfContours
        num_contours = glyph.numberOfContours

    if not coords or not endPts:
        return None

    w = max(c[0] for c in coords) - min(c[0] for c in coords)
    h = max(c[1] for c in coords) - min(c[1] for c in coords)
    ar = round(w / h, 1) if h else 0
    return (num_contours, len(coords), tuple(endPts), ar)


def build_glyph_to_unicode_map(tt: TTFont) -> dict:
    """
    Recursively decompiles OpenType GSUB ligatures into standard Unicode characters.
    Handles traditional and modern Malayalam ligature formations (Rachana, Meera, Manjari, etc.).
    """
    cmap = tt["cmap"].getBestCmap() if "cmap" in tt else {}
    rev_cmap = {v: chr(k) for k, v in cmap.items()}

    custom_bases = {
        "xx": "\u0D4D",        # chandrakkala / virama
        "r4": "\u0D4D\u0D30",  # ra sign prefix in GSUB
        "y2": "\u0D4D\u0D2F",  # ya sign
        "v2": "\u0D4D\u0D35",  # va sign
        "l2": "\u0D4D\u0D32",  # la sign
        "r2": "\u0D4D\u0D30",  # ra sign
        "u1": "\u0D41",        # u vowel sign
        "u2": "\u0D42",        # uu vowel sign
        "r1": "\u0D43",        # vocalic r
        "rh": "\u0D31",        # Malayalam letter RRA
        "rhrh": "\u0D31\u0D4D\u0D31",  # റ്റ
    }
    for k, v in custom_bases.items():
        if k not in rev_cmap:
            rev_cmap[k] = v

    gsub = tt.get("GSUB")
    if not gsub:
        return rev_cmap

    table = gsub.table
    ligs = {}
    for lookup in table.LookupList.Lookup:
        if lookup.LookupType == 4:  # Ligature substitution
            for subtable in lookup.SubTable:
                for first_glyph, lig_list in subtable.ligatures.items():
                    for lig in lig_list:
                        lig_name = lig.LigGlyph
                        components = [first_glyph] + lig.Component
                        ligs[lig_name] = components

    # Iteratively resolve nested ligatures
    for _ in range(6):
        for lig_name, components in ligs.items():
            if lig_name in rev_cmap:
                continue
            resolved_comps = []
            can_resolve = True
            for c in components:
                if c in rev_cmap:
                    resolved_comps.append(rev_cmap[c])
                else:
                    can_resolve = False
                    break
            if can_resolve:
                if components[0] == "r4" and len(components) == 2:
                    res = resolved_comps[1] + "\u0D4D\u0D30"
                else:
                    res = "".join(resolved_comps)
                rev_cmap[lig_name] = res

    return rev_cmap


def load_master_hashes() -> tuple[dict, dict]:
    """Loads all pre-indexed glyph hashes and builds topological fallback index."""
    global _MASTER_HASHES, _TOPO_SIGNATURES
    if _MASTER_HASHES is not None and _TOPO_SIGNATURES is not None:
        return _MASTER_HASHES, _TOPO_SIGNATURES

    _MASTER_HASHES = {}
    _TOPO_SIGNATURES = {}

    # 1. Load Universal Hashes (12,900+ entries)
    if os.path.exists(UNIVERSAL_HASH_MAP):
        try:
            with open(UNIVERSAL_HASH_MAP, "r", encoding="utf-8") as f:
                data = json.load(f)
                _MASTER_HASHES.update(data.get("glyph_hashes", {}))
                for k, v in data.get("topological_signatures", {}).items():
                    _TOPO_SIGNATURES[k] = v
        except Exception as e:
            print(f"[Font Mapper] Error loading universal hashes: {e}")

    # 2. Seed from Rachana/Meera json if needed
    for seed in (RACHANA_HASH_MAP, MEERA_HASH_MAP):
        if os.path.exists(seed):
            try:
                with open(seed, "r", encoding="utf-8") as f:
                    _MASTER_HASHES.update(json.load(f))
            except Exception:
                pass

    # 3. If master hashes empty or missing, decompile reference fonts dynamically
    if len(_MASTER_HASHES) < 1000:
        font_candidates = []
        if os.path.exists(REFERENCE_FONTS_DIR):
            for fname in os.listdir(REFERENCE_FONTS_DIR):
                if fname.lower().endswith(".ttf"):
                    font_candidates.append(os.path.join(REFERENCE_FONTS_DIR, fname))
        if os.path.exists(RACHANA_FONT_PATH) and RACHANA_FONT_PATH not in font_candidates:
            font_candidates.append(RACHANA_FONT_PATH)

        for font_path in font_candidates:
            try:
                ref_tt = TTFont(font_path)
                if 'glyf' not in ref_tt:
                    continue
                ref_glyf = ref_tt['glyf']
                g2u = build_glyph_to_unicode_map(ref_tt)

                for gname in ref_tt.getGlyphOrder():
                    if gname in ref_glyf and gname in g2u:
                        rg = ref_glyf[gname]
                        h = hash_glyph_coordinates(rg, ref_glyf)
                        if h and h not in _MASTER_HASHES:
                            _MASTER_HASHES[h] = g2u[gname]

                        sig = get_glyph_topology_signature(rg, ref_glyf)
                        if sig:
                            sig_key = str(sig)
                            if sig_key not in _TOPO_SIGNATURES:
                                _TOPO_SIGNATURES[sig_key] = g2u[gname]
                            if sig not in _TOPO_SIGNATURES:
                                _TOPO_SIGNATURES[sig] = g2u[gname]
            except Exception as e:
                print(f"[Font Mapper] Error building hashes from {font_path}: {e}")

        # Save to UNIVERSAL_HASH_MAP for fast subsequent startup
        try:
            with open(UNIVERSAL_HASH_MAP, "w", encoding="utf-8") as f:
                json.dump({"glyph_hashes": _MASTER_HASHES, "topological_signatures": {str(k): v for k, v in _TOPO_SIGNATURES.items()}}, f, ensure_ascii=False)
        except Exception as e:
            print(f"[Font Mapper] Could not save universal hashes: {e}")

    print(f"[Font Mapper] Loaded {len(_MASTER_HASHES)} master hashes and {len(_TOPO_SIGNATURES)} topological signatures.")
    return _MASTER_HASHES, _TOPO_SIGNATURES


def is_indic_font(font_name: str, font_buffer: bytes = None) -> bool:
    """Checks whether the font name or buffer corresponds to a Malayalam/Indic font."""
    from .font_installer import get_known_font_families
    known_families = get_known_font_families()

    # 1. Match against known family names
    if any(fam.lower() in font_name.lower() for fam in known_families):
        return True

    # 2. General Indic keywords
    generic_keywords = ("malayalam", "indic", "kerala", "karthika", "revathi", "samvad", "ism", "fml")
    if any(kw in font_name.lower() for kw in generic_keywords):
        return True

    # 3. Buffer inspection: check if font has cmap in Malayalam range (0x0D00 - 0x0D7F)
    if font_buffer:
        try:
            with tempfile.NamedTemporaryFile(suffix=".ttf", delete=False) as tmp:
                tmp.write(font_buffer)
                tmp_path = tmp.name
            tt = TTFont(tmp_path)
            cmap = tt["cmap"].getBestCmap() if "cmap" in tt else {}
            os.remove(tmp_path)
            if any(0x0D00 <= code <= 0x0D7F for code in cmap.keys()):
                return True
        except Exception:
            pass

    return False


def build_font_map_from_buffer(buffer: bytes, basefont: str) -> dict:
    """
    Builds a GID/CID -> Unicode mapping for a font buffer using
    the master hash database and topological signatures.
    """
    master_hashes, topo_sigs = load_master_hashes()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".ttf") as tmp:
        tmp.write(buffer)
        tmp_path = tmp.name

    cid_to_unicode = {}
    try:
        font = TTFont(tmp_path)
        if 'glyf' not in font:
            return {}
        glyf = font['glyf']
        glyph_order = font.getGlyphOrder()

        for gid, gname in enumerate(glyph_order):
            if gname in ('.notdef', '.null', 'space'):
                continue
            if gname not in glyf:
                continue

            glyph = glyf[gname]
            # 1. Exact coordinate hash match
            h = hash_glyph_coordinates(glyph, glyf)
            if h and h in master_hashes:
                text = master_hashes[h]
                cid_to_unicode[gid] = text
                cid_to_unicode[str(gid)] = text
                continue

            # 2. Topological signature match (for minor version nudges)
            sig = get_glyph_topology_signature(glyph, glyf)
            if sig:
                sig_key = str(sig)
                if sig in topo_sigs:
                    text = topo_sigs[sig]
                    cid_to_unicode[gid] = text
                    cid_to_unicode[str(gid)] = text
                elif sig_key in topo_sigs:
                    text = topo_sigs[sig_key]
                    cid_to_unicode[gid] = text
                    cid_to_unicode[str(gid)] = text

    except Exception as e:
        print(f"[Font Mapper] Error indexing font {basefont}: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return cid_to_unicode


def get_or_create_font_map(doc: pymupdf.Document, page_idx: int) -> dict:
    """
    Detects all Indic/Malayalam subset fonts on a page and returns their
    CID/GID -> Unicode mapping.
    
    CRITICAL: Does NOT modify the PDF doc or strip /ToUnicode streams!
    The PDF document's built-in /ToUnicode is preserved for all base characters.
    """
    ensure_cache_dir()
    page = doc[page_idx]
    fonts = page.get_fonts(full=True)
    result = {}

    for font_info in fonts:
        xref = font_info[0]
        font_name = font_info[3]
        basefont = font_info[3]
        clean_name = basefont.split("+")[-1] if "+" in basefont else basefont

        # Extract font binary
        try:
            _, _, _, buffer = doc.extract_font(xref)
            if not buffer:
                continue
        except Exception as e:
            print(f"[Font Mapper] Failed to extract font {basefont}: {e}")
            continue

        # Only process Indic/Malayalam fonts
        if not is_indic_font(basefont, buffer) and not is_indic_font(clean_name, buffer):
            continue

        font_data_hash = hash_font_data(buffer)
        cache_file = os.path.join(CACHE_DIR, f"{font_data_hash}_map.json")

        cid_to_correct = None

        # Check cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as cf:
                    cid_to_correct = json.load(cf)
            except Exception:
                cid_to_correct = None

        if not cid_to_correct:
            # 1. Build map via shape matching & topological signature
            cid_to_correct = build_font_map_from_buffer(buffer, basefont)

            # 2. If shape matching yields low coverage (< 10 glyphs) and Gemini is active,
            # autonomously analyze the unrecognized font and install it permanently!
            enable_gemini = os.getenv("ENABLE_GEMINI", "true").lower() in ("true", "1", "yes")
            if (not cid_to_correct or len(cid_to_correct) < 10) and enable_gemini:
                try:
                    from .font_analyzer import analyze_and_install_font
                    print(f"[Font Mapper] Unrecognized font '{basefont}' detected. Running autonomous analysis and installation...", flush=True)
                    analysis_res = analyze_and_install_font(basefont, buffer)
                    if analysis_res.get("status") == "success":
                        cid_to_correct = analysis_res.get("mappings", {})
                        print(f"[Font Mapper] Successfully analyzed and installed font '{basefont}' with {len(cid_to_correct)} glyphs!", flush=True)
                except Exception as ex:
                    print(f"[Font Mapper] Autonomous font analysis failed: {ex}", flush=True)

            if cid_to_correct:
                try:
                    with open(cache_file, "w", encoding="utf-8") as cf:
                        json.dump(cid_to_correct, cf, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"[Font Mapper] Failed to write cache {cache_file}: {e}")

        if cid_to_correct:
            # Merge into result under font_name and clean_name
            for key in (font_name, basefont, clean_name):
                if key not in result:
                    result[key] = {}
                result[key].update(cid_to_correct)

    return result
