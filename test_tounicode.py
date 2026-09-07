import pikepdf
doc = pikepdf.Pdf.open('/home/anuragh/Downloads/144-2026 M_removed.pdf')
for page in doc.pages:
    if '/Resources' in page and '/Font' in page.Resources:
        for name, font in page.Resources.Font.items():
            if str(font.get('/BaseFont')) == '/UXAVQA+Rachana-Regular':
                if font.get('/ToUnicode'):
                    print(font.get('/ToUnicode').read_bytes().decode('utf-8'))
                    import sys; sys.exit(0)
