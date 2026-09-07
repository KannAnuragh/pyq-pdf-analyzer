import pikepdf, io, hashlib
from fontTools.ttLib import TTFont

def hash_glyph_coordinates_fixed(glyph, glyf_table):
    if glyph.isComposite():
        # Get expanded coordinates
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

doc = pikepdf.Pdf.open('/home/anuragh/Downloads/144-2026 M_removed.pdf')
font = doc.pages[0].Resources.Font['/f-2-1'].get('/DescendantFonts')[0]
data = font.get('/FontDescriptor').get('/FontFile2').read_bytes()
subset = TTFont(io.BytesIO(data))
glyf = subset['glyf']

for cid in [70, 75, 87, 111, 89, 112]:
    gname = subset.getGlyphOrder()[cid]
    glyph = glyf[gname]
    h = hash_glyph_coordinates_fixed(glyph, glyf)
    print(f'CID {cid} (ASCII {chr(cid)}): Hash {h}')
