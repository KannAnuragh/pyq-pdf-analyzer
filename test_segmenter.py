import sys, json
from pdf_qa_agent.ingest.router import extract_text_from_pdf
from pdf_qa_agent.ingest.layout_segmenter import segment_questions

res = extract_text_from_pdf('pdf_qa_agent/data/uploads/ed3a3b72-550a-4a1a-a3ac-8ed35fe422da_12CB CH-4 Moving charges & Magnetism (DPP).pdf')
span_blocks = res['span_blocks']

segments = segment_questions(span_blocks)
print(f"Segmenter found {len(segments)} segments")
for i, s in enumerate(segments):
    print(f"Segment {i+1}: Q{s.q_num}")
