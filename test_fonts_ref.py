import fitz
import glob
import os

pdf_files = glob.glob('backend/uploads/**/*.pdf', recursive=True)
if not pdf_files:
    pdf_files = glob.glob('**/*.pdf', recursive=True)

print('PDFs found:', pdf_files)

if pdf_files:
    pdf = pdf_files[0]
    print('Testing PDF:', pdf)
    doc = fitz.open(pdf)
    
    # Check first 5 pages
    for i in range(min(5, len(doc))):
        print(f"\n--- Page {i+1} Fonts ---")
        page = doc[i]
        fonts = page.get_fonts(full=True)
        for f in fonts:
            print(' -', f)
            
        print("\n--- Raw Text Snippet ---")
        print(page.get_text("text")[:100].replace('\n', ' '))
