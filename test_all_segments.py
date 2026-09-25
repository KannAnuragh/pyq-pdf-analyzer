import sys, json, os
import re
sys.path.insert(0, os.path.abspath('.'))

import pdf_qa_agent.ingest.router
from pdf_qa_agent.ingest.layout_segmenter import _Q_HEADER_RE, _is_question_boundary, QuestionSegment, LayoutElement

res = pdf_qa_agent.ingest.router.extract_text_from_pdf('pdf_qa_agent/data/uploads/ed3a3b72-550a-4a1a-a3ac-8ed35fe422da_12CB CH-4 Moving charges & Magnetism (DPP).pdf')
span_blocks = res['span_blocks']

segments = []
current_segment = None
for elem_dict in span_blocks:
    if "text" not in elem_dict: continue
    e_dict = elem_dict.copy()
    e_dict.pop('flags', None)
    
    elem = LayoutElement(**e_dict)
    text = elem.text.strip()
    
    q_m = _Q_HEADER_RE.match(text)
    if q_m:
        q_num = int(q_m.group(1))
        remainder = text[q_m.end():]
        gap_size = len(remainder) - len(remainder.lstrip())
        
        if 1 <= q_num <= 300 and _is_question_boundary(elem, 12.0, gap_size, None, current_segment):
            if current_segment is not None and current_segment.elements:
                segments.append(current_segment)
            current_segment = QuestionSegment(q_num=q_num, page=elem.page, section=None, elements=[elem])
            continue
            
    if current_segment is not None:
        current_segment.elements.append(elem)

if current_segment is not None and current_segment.elements:
    segments.append(current_segment)

print(f"Total raw segments: {len(segments)}")
for i, s in enumerate(segments):
    print(f"Raw Seg {i+1} (Q{s.q_num}): {s.flat_text[:70]!r}")
