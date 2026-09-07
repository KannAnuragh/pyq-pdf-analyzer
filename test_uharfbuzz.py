import uharfbuzz as hb
from fontTools.ttLib import TTFont
import hashlib

def hash_glyph_coordinates(glyph):
    if not hasattr(glyph, 'coordinates') or not glyph.coordinates:
        return None
    xMin = min(pt[0] for pt in glyph.coordinates)
    yMin = min(pt[1] for pt in glyph.coordinates)
    coords = []
    for (x, y) in glyph.coordinates:
        coords.extend([x - xMin, y - yMin])
    return hashlib.md5(str(coords).encode()).hexdigest()

font_path = '/home/anuragh/Trogon/qpaper scanner/pdf_qa_agent/rachana/ttf/Rachana-Regular.ttf'
blob = hb.Blob.from_file_path(font_path)
face = hb.Face(blob)
font = hb.Font(face)
ttfont = TTFont(font_path)
glyf = ttfont['glyf']

def get_glyph_hash_for_text(text):
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(font, buf)
    
    infos = buf.glyph_infos
    gids = [info.codepoint for info in infos]
    print(f"'{text}' shapes to GIDs:", gids)
    
    hashes = []
    for gid in gids:
        gname = ttfont.getGlyphName(gid)
        glyph = glyf[gname]
        h = hash_glyph_coordinates(glyph)
        hashes.append(h)
    return hashes

get_glyph_hash_for_text('ഗു')
get_glyph_hash_for_text('പൂ')
get_glyph_hash_for_text('നു')
