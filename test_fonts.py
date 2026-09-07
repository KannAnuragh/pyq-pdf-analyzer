import pikepdf
doc = pikepdf.Pdf.open('/home/anuragh/Downloads/144-2026 M_removed.pdf')
for page in doc.pages:
    if '/Resources' in page and '/Font' in page.Resources:
        for name, font in page.Resources.Font.items():
            print('Font', name, ':', font.get('/BaseFont'))
            if font.get('/Subtype') == '/Type0':
                desc = font.get('/DescendantFonts')[0]
                print('  CIDToGIDMap:', desc.get('/CIDToGIDMap'))
                print('  CIDSystemInfo:', desc.get('/CIDSystemInfo'))
            if font.get('/ToUnicode'):
                print('  Has ToUnicode')
