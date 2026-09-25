import sys, os, json
sys.path.insert(0, os.path.abspath('.'))
from graph.nodes import extract_node
from ingest.router import extract_text_from_pdf

res = extract_text_from_pdf('data/uploads/ed3a3b72-550a-4a1a-a3ac-8ed35fe422da_12CB CH-4 Moving charges & Magnetism (DPP).pdf')
state = {'raw_text': res['full_text'], 'pages': res['pages'], 'span_blocks': res['span_blocks'], 'token_usage': {}}
out = extract_node(state)
for q in out['extracted_blocks']:
    print(f"Q{q.get('question_number')}: {q.get('stem', '')}")
