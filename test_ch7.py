import sys
sys.path.insert(0, '/app/pdf_qa_agent')
from ingest.router import route_and_extract
res = route_and_extract('data/uploads/ch7.pdf')
for q in res.get('extracted_blocks', []):
    print(f"Q{q.get('question_number')}: {repr(q.get('stem', ''))[:60]}...")
