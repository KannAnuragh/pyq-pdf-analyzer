import pymupdf, re

def test():
    doc = pymupdf.open('data/uploads/6e0d64d9-e3ee-4df2-86ac-696d267766fb_CH-1 Electric Charges & Field (DPP).pdf')
    page = doc[1]
    
    from pdf_qa_agent.ingest.router import _rebuild_text_from_font_map
    text, unmapped = _rebuild_text_from_font_map(page, font_map={})
    print("CURRENT OUTPUT:")
    for line in text.split('\n'):
        if '1.5' in line or '3 N' in line or 'densities' in line:
            print(" ", repr(line))

test()
