import sys
import os
import json

sys.path.insert(0, "/home/anuragh/Trogon/qpaper scanner")

from pdf_qa_agent.ingest.router import extract_text_from_pdf
from pdf_qa_agent.graph.nodes import extract_node

def test_pdf(pdf_path, name):
    print(f"\n=======================================================")
    print(f"Testing {name}: {pdf_path}")
    print(f"=======================================================")
    
    ingest_res = extract_text_from_pdf(pdf_path)
    state = {
        "raw_text": ingest_res["full_text"],
        "pages": ingest_res["pages"],
        "span_blocks": ingest_res["span_blocks"],
        "document_id": pdf_path,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }
    
    result = extract_node(state)
    questions = result.get("extracted_blocks", [])
    print(f"\nTotal questions extracted: {len(questions)}")
    
    for q in questions:
        q_num = q.get("question_number")
        stem = (q.get("stem") or "").replace("\n", " ")[:60]
        opts = q.get("options") or {}
        ans = q.get("marked_answer")
        opt_keys = list(opts.keys())
        print(f"Q{q_num:2d} | Ans: {ans or '-'} | Opts: {opt_keys} | Stem: {stem}")
        if len(opts) < 4:
            print(f"     WARNING: Opt count is {len(opts)}: {opts}")
            
    return questions

if __name__ == "__main__":
    p1 = "/home/anuragh/Trogon/qpaper scanner/pdf_qa_agent/data/uploads/7745b22d-2c25-42b7-884a-6f663d3ed5dd_MATHS.pdf"
    p2 = "/home/anuragh/Trogon/qpaper scanner/pdf_qa_agent/data/uploads/24e674e5-56bb-419e-8422-dc33c20e9c86_MATHS_JEST_with_answer_key.pdf"
    
    test_pdf(p1, "MATHS.pdf")
    test_pdf(p2, "MATHS_JEST_with_answer_key.pdf")
