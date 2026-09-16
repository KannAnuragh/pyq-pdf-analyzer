import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pdf_qa_agent.storage.config_manager import get_setting, get_all_settings, update_settings
from pdf_qa_agent.graph.nodes import parse_questions_from_text
from pdf_qa_agent.ingest.router import ocr_whole_pdf

def test_settings():
    print("Testing config_manager settings...")
    settings = get_all_settings()
    assert "auto_whole_pdf_ocr_threshold" in settings, "Missing auto_whole_pdf_ocr_threshold"
    assert "auto_whole_pdf_ocr_engine" in settings, "Missing auto_whole_pdf_ocr_engine"
    assert settings["auto_whole_pdf_ocr_threshold"] == 3
    print("Settings check passed! Threshold:", settings["auto_whole_pdf_ocr_threshold"])

def test_streak_detector():
    print("Testing streak detection logic...")
    streak_threshold = 3

    def detect_streak(fallbacks):
        sorted_fallbacks = sorted(set(fallbacks))
        longest_streak = []
        cur_streak = []
        for qn in sorted_fallbacks:
            if not cur_streak or qn == cur_streak[-1] + 1:
                cur_streak.append(qn)
                if len(cur_streak) > len(longest_streak):
                    longest_streak = list(cur_streak)
            else:
                cur_streak = [qn]
        return longest_streak if len(longest_streak) >= streak_threshold else []

    # Case 1: 2 consecutive (below threshold)
    assert len(detect_streak([1, 2, 5, 8])) == 0, "Should not trigger on streak of 2"
    
    # Case 2: 3 consecutive (at threshold)
    assert detect_streak([1, 2, 3, 7]) == [1, 2, 3], "Should trigger on streak of 3"

    # Case 3: 4 consecutive scattered in list
    assert detect_streak([1, 4, 5, 6, 7, 10]) == [4, 5, 6, 7], "Should find streak of 4"

    print("Streak detection logic passed!")

def test_parse_questions():
    print("Testing parse_questions_from_text...")
    sample_text = """
1. What is the capital of Kerala?
A. Kochi
B. Kozhikode
C. Thiruvananthapuram
D. Kollam
Correct Answer: C

2. What is 2 + 2?
(A) 3
(B) 4
(C) 5
(D) 6
Marked Answer: B
"""
    questions = parse_questions_from_text(sample_text)
    assert len(questions) == 2, f"Expected 2 questions, got {len(questions)}"
    assert questions[0]["question_number"] == 1
    assert questions[1]["question_number"] == 2
    assert "Thiruvananthapuram" in questions[0]["options"]["C"]
    print("parse_questions_from_text check passed!")

def test_repair_node_streak():
    print("Testing repair_node consecutive streak escalation...")
    from unittest.mock import patch
    from pdf_qa_agent.graph.nodes import repair_node
    from pdf_qa_agent.graph.state import GraphState

    # Construct mock state with 5 questions, where Q1, Q2, Q3 are unreadable
    blocks = [
        {"question_number": 1, "stem": "Garbled text 1", "options": {}},
        {"question_number": 2, "stem": "Garbled text 2", "options": {}},
        {"question_number": 3, "stem": "Garbled text 3", "options": {}},
        {"question_number": 4, "stem": "Clean stem 4", "options": {"A": "1", "B": "2"}},
        {"question_number": 5, "stem": "Clean stem 5", "options": {"A": "1", "B": "2"}},
    ]
    quality_scores = {
        1: {"status": "unreadable", "coverage_pct": 0.2},
        2: {"status": "unreadable", "coverage_pct": 0.3},
        3: {"status": "unreadable", "coverage_pct": 0.1},
        4: {"status": "ok", "coverage_pct": 0.98},
        5: {"status": "ok", "coverage_pct": 0.99},
    }
    state = {
        "file_path": "/tmp/dummy_test.pdf",
        "raw_text": "dummy raw text",
        "pages": [],
        "extracted_blocks": blocks,
        "quality_scores": quality_scores,
        "repair_log": [],
        "vision_calls_count": 0,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    mock_ocr_return = {
        "full_text": "1. What is capital?\nA. A\nB. B\nCorrect Answer: A\n\n2. Q2?\nA. 1\nB. 2\nCorrect Answer: B\n\n3. Q3?\nA. 3\nB. 4\nCorrect Answer: A",
        "pages": [{"page_num": 0, "text": "...", "font_hashes": {}, "char_confidences": {}, "unmapped_codes": [], "validation_flags": [], "has_text_layer": True}],
        "prompt_tokens": 50,
        "completion_tokens": 20,
        "total_tokens": 70,
    }

    with patch("os.path.exists", return_value=True):
        with patch("pdf_qa_agent.ingest.router.ocr_whole_pdf", return_value=mock_ocr_return) as mock_ocr:
            res = repair_node(state)
            assert mock_ocr.called, "ocr_whole_pdf should have been called when streak >= 3!"
            assert res is not None
            assert len(res["extracted_blocks"]) == 3, f"Expected 3 recovered blocks, got {len(res['extracted_blocks'])}"
            assert any(entry.get("action") == "ocr_whole_pdf_escalation" for entry in res["repair_log"]), "Repair log missing escalation action"
            print("repair_node streak escalation successfully triggered and verified!")

def test_correction_node_streak():
    print("Testing correction_node consecutive streak OCR escalation...")
    from unittest.mock import patch, MagicMock
    from pdf_qa_agent.graph.correction_node import correction_and_validation_node, QuestionCorrection

    blocks = [
        {"question_number": 1, "stem": "Garbled stem 1", "options": {"A": "1", "B": "2"}, "marked_answer": "A"},
        {"question_number": 2, "stem": "Garbled stem 2", "options": {"A": "1", "B": "2"}, "marked_answer": "B"},
        {"question_number": 3, "stem": "Garbled stem 3", "options": {"A": "1", "B": "2"}, "marked_answer": "A"},
    ]
    state = {
        "file_path": "/tmp/dummy_test.pdf",
        "extracted_blocks": blocks,
        "pages": [{"font_hashes": {}, "char_confidences": {}}],
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    mock_corrections = [
        QuestionCorrection(question_number=1, needs_ocr_fallback=True, validated_answer="A", answer_status="agrees"),
        QuestionCorrection(question_number=2, needs_ocr_fallback=True, validated_answer="B", answer_status="agrees"),
        QuestionCorrection(question_number=3, needs_ocr_fallback=True, validated_answer="A", answer_status="agrees"),
    ]

    mock_ocr_return = {
        "full_text": "1. Recovered 1\nA. 1\nB. 2\nCorrect Answer: A\n\n2. Recovered 2\nA. 1\nB. 2\nCorrect Answer: B\n\n3. Recovered 3\nA. 1\nB. 2\nCorrect Answer: A",
        "pages": [{"page_num": 0, "text": "...", "font_hashes": {}, "char_confidences": {}, "unmapped_codes": [], "validation_flags": [], "has_text_layer": True}],
        "prompt_tokens": 50,
        "completion_tokens": 20,
        "total_tokens": 70,
    }

    with patch("pdf_qa_agent.graph.correction_node.get_correction_provider", return_value="groq"), \
         patch("pdf_qa_agent.graph.correction_node.is_word_correction_enabled", return_value=False), \
         patch("pdf_qa_agent.graph.correction_node._call_llm_batched", return_value=(mock_corrections, 10, 10)), \
         patch("pdf_qa_agent.graph.correction_node._apply_post_processing_gate", return_value=([], [], [1, 2, 3])), \
         patch("os.path.exists", return_value=True), \
         patch("pdf_qa_agent.ingest.router.ocr_whole_pdf", return_value=mock_ocr_return) as mock_ocr, \
         patch("pdf_qa_agent.graph.nodes._solve_single_block", return_value={"validated_answer": "A", "is_valid": True, "reasoning": "Agreed"}):
        res = correction_and_validation_node(state)
        assert mock_ocr.called, "ocr_whole_pdf should have been called in correction_node Step 5.5!"
        assert res is not None
        # Check stem was updated from recovered text
        assert res["extracted_blocks"][0]["stem"] == "Recovered 1"
        print("correction_node streak OCR escalation successfully triggered and verified!")

if __name__ == "__main__":
    test_settings()
    test_streak_detector()
    test_parse_questions()
    test_repair_node_streak()
    test_correction_node_streak()
    print("ALL TESTS PASSED SUCCESSFULLY!")
