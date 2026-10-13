import sys
import os
import glob
import traceback

sys.path.insert(0, "/home/anuragh/Trogon/qpaper scanner")

from pdf_qa_agent.ingest.router import extract_text_from_pdf
from pdf_qa_agent.graph.nodes import extract_node

def test_single_pdf(file_path, display_name):
    print(f"\n{'='*70}")
    print(f"TESTING: {display_name}")
    print(f"Path: {file_path}")
    print(f"{'='*70}")
    
    try:
        res = extract_text_from_pdf(file_path)
        print(f"[Ingest] Pages: {len(res['pages'])}, Raw text len: {len(res['full_text'])}, Span blocks: {len(res.get('span_blocks', []))}")
        
        state = {
            "raw_text": res["full_text"],
            "pages": res["pages"],
            "span_blocks": res["span_blocks"],
            "document_id": file_path,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        
        out = extract_node(state)
        questions = out.get("extracted_blocks", [])
        print(f"[Extract] Total extracted: {len(questions)}")
        
        if not questions:
            print("  ⚠️ WARNING: 0 questions extracted!")
            return False
            
        # Summary statistics
        mcq_count = sum(1 for q in questions if q.get("question_type") == "mcq")
        has_ans_count = sum(1 for q in questions if q.get("marked_answer"))
        full_opts_count = sum(1 for q in questions if len(q.get("options") or {}) >= 4)
        has_stem_count = sum(1 for q in questions if bool(q.get("stem") and q.get("stem").strip()))
        
        print(f"  Summary: Total={len(questions)}, MCQs={mcq_count}, Full 4-options={full_opts_count}, With marked_answer={has_ans_count}, With stem={has_stem_count}")
        
        # Check first 3 and last 2
        sample_indices = list(range(min(3, len(questions)))) + list(range(max(0, len(questions)-2), len(questions)))
        seen_idx = set()
        for idx in sample_indices:
            if idx in seen_idx or idx >= len(questions):
                continue
            seen_idx.add(idx)
            q = questions[idx]
            q_num = q.get("question_number")
            q_type = q.get("question_type")
            stem = (q.get("stem") or "").replace("\n", " ")[:65]
            opts = list((q.get("options") or {}).keys())
            ans = q.get("marked_answer") or "-"
            print(f"  Q{q_num:2d} [{q_type}] Ans={ans} Opts={opts} Stem: {stem}")
            
        return True
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        traceback.print_exc()
        return False

def run_suite():
    uploads_dir = "/home/anuragh/Trogon/qpaper scanner/pdf_qa_agent/data/uploads"
    
    # Pick representative PDFs from each category
    targets = [
        ("MATHS.pdf", "Mathematics Standard (Cambria CID font)"),
        ("MATHS_JEST_with_answer_key.pdf", "Mathematics JEST Bilingual (NotoSansMalayalam)"),
        ("ch7.pdf", "Physics CBSE DPP (Alternating Current with Diagrams/Formulas)"),
        ("CHEMISTRY_JEST_with_option_key.pdf", "Chemistry JEST with option key"),
        ("BIOLOGY.pdf", "Biology Exam"),
        ("6 CBSE Chapter 03 Landforms and Life Practise Exam.pdf", "CBSE Social Science Practice Exam"),
        ("Carpenter - Kerala state water Transport.pdf", "Kerala PSC Legacy Indic Malayalam"),
        ("MentorBee_LMS_Question_Format_Sample.pdf", "LMS Question Format Sample")
    ]
    
    results = {}
    for filename_substr, desc in targets:
        matches = glob.glob(os.path.join(uploads_dir, f"*{filename_substr}*"))
        if not matches:
            print(f"Could not find matching PDF for {filename_substr}")
            continue
        # Use first match
        pdf_path = matches[0]
        ok = test_single_pdf(pdf_path, desc)
        results[desc] = ok

    print("\n" + "="*70)
    print("TEST SUITE SUMMARY:")
    for desc, ok in results.items():
        status = "PASSED ✓" if ok else "FAILED ❌"
        print(f"  {status}: {desc}")
    print("="*70)

if __name__ == "__main__":
    run_suite()
