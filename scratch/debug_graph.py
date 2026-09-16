import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pdf_qa_agent.graph.nodes import extract_node, quality_score_node, repair_node
from pdf_qa_agent.graph.correction_node import correction_and_validation_node, get_correction_provider
from pdf_qa_agent.graph.state import GraphState
from pdf_qa_agent.storage import database
import logging

logging.basicConfig(level=logging.INFO)

db = database.SessionLocal()
try:
    print(f"Correction provider: {get_correction_provider()}")
    # Read the text dump
    with open("raw_extracted.txt", "r") as f:
        text = f.read()

    state = {
        "raw_text": text,
        "file_path": "test_job",
        "pages": [{"page_num": 1, "font_hashes": {}, "char_confidences": {}}],
        "extracted_blocks": [],
        "quality_scores": {},
        "repair_log": [],
        "vision_calls_count": 0,
        "processed_questions": [],
        "summary": {"total": 0, "validated": 0, "flagged": 0},
        "token_usage": {}
    }

    state.update(extract_node(state))
    state.update(quality_score_node(state))
    state.update(repair_node(state))
    state.update(correction_and_validation_node(state))
    
    processed = state.get("processed_questions", [])
    print(f"Processed {len(processed)} questions.")
    if processed:
        print("First processed question:")
        for k, v in processed[0].items():
            if k in ["stem", "options"]: continue
            print(f"  {k}: {v}")
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    db.close()
