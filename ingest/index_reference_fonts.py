"""
Universal Malayalam Reference Font Indexer.

Scans all reference fonts in storage/reference_fonts/, performs HarfBuzz shaping
across consonants, vowel signs, chillus, and conjuncts, hashes all glyph outlines,
and produces a unified universal_glyph_hashes.json mapping:
    outline_hash -> Unicode Malayalam text
"""

import os
import json
import hashlib
import uharfbuzz as hb
from fontTools.ttLib import TTFont

REFERENCE_FONTS_DIR = os.path.join(os.path.dirname(__file__), "..", "storage", "reference_fonts")
LEGACY_MAPS_DIR = os.path.join(os.path.dirname(__file__), "..", "storage", "legacy_maps")
OUTPUT_HASH_MAP = os.path.join(os.path.dirname(__file__), "..", "storage", "universal_glyph_hashes.json")


def hash_glyph_coordinates(glyph, glyf_table):
    """Creates a stable MD5 hash for a glyph based on normalized contour coordinates."""
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


def build_shaping_combinations():
    """Builds an exhaustive list of Malayalam text sequences to shape."""
    consonants = [chr(c) for c in range(0x0D15, 0x0D3A)]
    consonants = [c for c in consonants if c.isprintable() and c.strip()]
    vowel_signs = [chr(c) for c in range(0x0D3E, 0x0D4E)]
    virama = chr(0x0D4D)
    chillus = [chr(c) for c in range(0x0D7A, 0x0D80)]
    independent_vowels = [chr(c) for c in range(0x0D05, 0x0D15)]
    digits = [chr(c) for c in range(0x0D66, 0x0D70)]
    signs = [chr(0x0D02), chr(0x0D03)]  # anusvara, visarga

    combinations = []
    combinations.extend(consonants)
    combinations.extend(chillus)
    combinations.extend(independent_vowels)
    combinations.extend(digits)
    combinations.extend(signs)

    # Consonant + vowel signs & anusvara/visarga
    for c in consonants:
        for v in vowel_signs + signs:
            combinations.append(c + v)

    # 2-letter conjuncts
    for c1 in consonants:
        for c2 in consonants:
            combinations.append(c1 + virama + c2)
            # Common vowel signs on 2-letter conjuncts
            for v in ["\u0D3E", "\u0D3F", "\u0D40", "\u0D41", "\u0D42", "\u0D46", "\u0D47", "\u0D48", "\u0D4A", "\u0D4B"]:
                combinations.append(c1 + virama + c2 + v)

    # 3-letter conjuncts (most common combinations like ന്ത്ര, ങ്ക്യ, ഷ്ട്ര, ന്ത്യ, ത്സ്യ, ക്ഷ്മ, etc.)
    common_third = ["\u0D2F", "\u0D30", "\u0D32", "\u0D35"]  # ya, ra, la, va
    for c1 in consonants:
        for c2 in consonants:
            for c3 in common_third:
                combinations.append(c1 + virama + c2 + virama + c3)

    return combinations


def index_unicode_font(font_path: str, master_map: dict):
    """Indexes all direct cmap glyphs and HarfBuzz shaped ligatures for a Unicode font."""
    font_name = os.path.basename(font_path)
    try:
        ttfont = TTFont(font_path)
        if 'glyf' not in ttfont:
            print(f"[Indexer] Skipping {font_name} (no glyf table)")
            return 0
        glyf = ttfont['glyf']
    except Exception as e:
        print(f"[Indexer] Error reading {font_name}: {e}")
        return 0

    added = 0
    # 1. Direct cmap entries
    cmap = ttfont['cmap'].getBestCmap() if 'cmap' in ttfont else {}
    for uni_int, gname in cmap.items():
        if gname in glyf:
            glyph = glyf[gname]
            h = hash_glyph_coordinates(glyph, glyf)
            if h and h not in master_map:
                master_map[h] = chr(uni_int)
                added += 1

    # 2. HarfBuzz shaping for conjunct ligatures
    try:
        blob = hb.Blob.from_file_path(font_path)
        face = hb.Face(blob)
        font = hb.Font(face)
    except Exception as e:
        print(f"[Indexer] HarfBuzz error on {font_name}: {e}")
        return added

    combinations = build_shaping_combinations()
    for text in combinations:
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(font, buf)

        infos = buf.glyph_infos
        if len(infos) == 1:
            gid = infos[0].codepoint
            gname = ttfont.getGlyphName(gid)
            if gname in glyf:
                glyph = glyf[gname]
                h = hash_glyph_coordinates(glyph, glyf)
                if h and h not in master_map:
                    master_map[h] = text
                    added += 1

    print(f"[Indexer] {font_name}: added {added} new unique glyph hashes")
    return added


def index_legacy_font(font_path: str, map_path: str, master_map: dict):
    """Indexes glyphs from a legacy font using its corresponding .map conversion file."""
    font_name = os.path.basename(font_path)
    if not os.path.exists(font_path) or not os.path.exists(map_path):
        return 0

    try:
        ttfont = TTFont(font_path)
        if 'glyf' not in ttfont:
            return 0
        glyf = ttfont['glyf']
        cmap = ttfont['cmap'].getBestCmap() if 'cmap' in ttfont else {}
    except Exception as e:
        print(f"[Indexer] Error loading legacy font {font_name}: {e}")
        return 0

    # Parse .map file
    char_to_uni = {}
    with open(map_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            parts = line.split("=", 1)
            char_to_uni[parts[0]] = parts[1]

    added = 0
    # Map single-character keys
    for char, uni in char_to_uni.items():
        if len(char) == 1:
            ord_val = ord(char)
            if ord_val in cmap:
                gname = cmap[ord_val]
                if gname in glyf:
                    h = hash_glyph_coordinates(glyf[gname], glyf)
                    if h and h not in master_map:
                        master_map[h] = uni
                        added += 1

    print(f"[Indexer] Legacy {font_name} via {os.path.basename(map_path)}: added {added} unique hashes")
    return added


def generate_universal_index():
    master_map = {}

    # Seed with existing Rachana and Meera maps if present
    base_dir = os.path.dirname(__file__)
    for seed_file in ["rachana_hash_map.json", "meera_hash_map.json"]:
        seed_path = os.path.join(base_dir, seed_file)
        if os.path.exists(seed_path):
            try:
                with open(seed_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    master_map.update(data)
                    print(f"[Indexer] Seeded {len(data)} hashes from {seed_file}")
            except Exception as e:
                print(f"[Indexer] Failed to load seed {seed_file}: {e}")

    # Index all Unicode fonts in REFERENCE_FONTS_DIR
    if os.path.exists(REFERENCE_FONTS_DIR):
        for fname in sorted(os.listdir(REFERENCE_FONTS_DIR)):
            if not fname.endswith(".ttf"):
                continue
            font_path = os.path.join(REFERENCE_FONTS_DIR, fname)
            if "ML-TT" in fname or "Karthika" in fname or "Revathi" in fname:
                # Handle legacy font
                karthika_map = os.path.join(LEGACY_MAPS_DIR, "karthika.map")
                index_legacy_font(font_path, karthika_map, master_map)
            else:
                index_unicode_font(font_path, master_map)

    print(f"\n[Indexer] Total master glyph hashes compiled: {len(master_map)}")
    with open(OUTPUT_HASH_MAP, "w", encoding="utf-8") as f:
        json.dump(master_map, f, ensure_ascii=False, indent=2)
    print(f"[Indexer] Written master database to {OUTPUT_HASH_MAP}")
    return len(master_map)


if __name__ == "__main__":
    generate_universal_index()
