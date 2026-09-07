import uharfbuzz as hb
from fontTools.ttLib import TTFont
import hashlib
import json
import itertools

def hash_glyph_coordinates(glyph, glyf_table):
    if glyph.isComposite():
        coords, endPts, flags = glyph.getCoordinates(glyf_table)
    else:
        if not hasattr(glyph, 'coordinates') or not glyph.coordinates: return None
        coords = glyph.coordinates

    xMin = min(pt[0] for pt in coords)
    yMin = min(pt[1] for pt in coords)
    normalized = []
    for (x, y) in coords:
        normalized.extend([x - xMin, y - yMin])
    return hashlib.md5(str(normalized).encode()).hexdigest()

font_path = '/home/anuragh/Trogon/qpaper scanner/pdf_qa_agent/rachana/ttf/Meera-Regular.ttf'
blob = hb.Blob.from_file_path(font_path)
face = hb.Face(blob)
font = hb.Font(face)
ttfont = TTFont(font_path)
glyf = ttfont['glyf']

hash_to_text = {}

consonants = [chr(c) for c in range(0x0D15, 0x0D3A)]
consonants = [c for c in consonants if c.isprintable() and c.strip()]
vowel_signs = [chr(c) for c in range(0x0D3E, 0x0D4E)]
virama = chr(0x0D4D)
chillus = [chr(c) for c in range(0x0D7A, 0x0D80)]

combinations = []
combinations.extend(consonants)
combinations.extend(chillus)
for c in consonants:
    for v in vowel_signs:
        combinations.append(c + v)
for c1 in consonants:
    for c2 in consonants:
        combinations.append(c1 + virama + c2)
        for v in vowel_signs:
            combinations.append(c1 + virama + c2 + v)
for c1, c2, c3 in itertools.product(consonants, repeat=3):
    combinations.append(c1 + virama + c2 + virama + c3)

print(f"Generating hashes for {len(combinations)} combinations...")

for text in combinations:
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf)
    
    infos = buf.glyph_infos
    if len(infos) == 1:
        gid = infos[0].codepoint
        gname = ttfont.getGlyphName(gid)
        glyph = glyf[gname]
        h = hash_glyph_coordinates(glyph, glyf)
        if h and h not in hash_to_text:
            hash_to_text[h] = text

print(f"Successfully mapped {len(hash_to_text)} unique glyphs!")
with open('ingest/meera_hash_map.json', 'w') as f:
    json.dump(hash_to_text, f, ensure_ascii=False, indent=2)
