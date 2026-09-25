import sys, json, os
sys.path.insert(0, os.path.abspath('.'))

import pdf_qa_agent.ingest.router

res = pdf_qa_agent.ingest.router.extract_text_from_pdf('pdf_qa_agent/data/uploads/ed3a3b72-550a-4a1a-a3ac-8ed35fe422da_12CB CH-4 Moving charges & Magnetism (DPP).pdf')
span_blocks = res['span_blocks']

for elem in span_blocks:
    if "text" not in elem: continue
    text = elem["text"].strip()
    if elem["page"] == 0:
        print(f"{text!r} | bbox={elem['bbox']} | y_gap={(elem['bbox'][1] - prev_bbox[3]) if 'prev_bbox' in locals() else 'N/A'}")
        prev_bbox = elem['bbox']
        if text.startswith('1. A current'):
            break
