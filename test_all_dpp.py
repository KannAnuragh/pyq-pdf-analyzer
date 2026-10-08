import sys
import os
from pdf_qa_agent.ingest.router import route_and_extract

dpp_dir = "/data/uploads/dpp"
if not os.path.exists(dpp_dir):
    print("DPP dir not found in container")
    sys.exit(0)

pdfs = [f for f in os.listdir(dpp_dir) if f.endswith('.pdf')]
for pdf in sorted(pdfs):
    res = route_and_extract(os.path.join(dpp_dir, pdf))
    qs = res.get('extracted_blocks', [])
    q_nums = [q.get('question_number') for q in qs]
    print(f"{pdf}: {len(qs)} questions extracted. Numbers: {q_nums[:5]} ... {q_nums[-5:] if len(q_nums) > 5 else ''}")
