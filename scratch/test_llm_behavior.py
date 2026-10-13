import sys
import os

# Add repo root to python path
sys.path.insert(0, os.path.abspath("."))

from pdf_qa_agent.graph.correction_node import (
    QuestionCorrection,
    DirectQuestionCorrection,
    _apply_post_processing_gate,
    CORRECTION_SYSTEM_PROMPT,
    DIRECT_CORRECTION_SYSTEM_PROMPT
)

def test_pydantic_schema_and_defaults():
    # Test QuestionCorrection with unanswered descriptive question
    qc = QuestionCorrection(
        question_number=1,
        question_type="descriptive_single",
        corrected_stem="Explain the photoelectric effect.",
        corrected_options={},
        validated_answer=None,
        answer_status="none",
        corrected_explanation=None,
        needs_ocr_fallback=False,
        is_too_messy=False
    )
    assert qc.validated_answer is None
    assert qc.corrected_explanation is None
    assert qc.question_type == "descriptive_single"
    print("✓ QuestionCorrection schema test passed")

    # Test DirectQuestionCorrection
    dqc = DirectQuestionCorrection(
        question_number=2,
        is_too_messy=True,
        needs_ocr_fallback=True,
        question_type="mcq",
        corrected_stem="Raw math with circuit diagram",
        corrected_options={},
        validated_answer=None,
        answer_status="none",
        corrected_explanation=None
    )
    assert dqc.is_too_messy is True
    assert dqc.needs_ocr_fallback is True
    print("✓ DirectQuestionCorrection schema test passed")

def test_ocr_fallback_gate():
    # Question 1: needs OCR fallback because of diagram requirement
    c1 = QuestionCorrection(
        question_number=1,
        needs_ocr_fallback=True,
        question_type="descriptive_single"
    )
    # Question 2: normal clean MCQ
    c2 = QuestionCorrection(
        question_number=2,
        needs_ocr_fallback=False,
        is_too_messy=False,
        question_type="mcq",
        validated_answer="B",
        answer_status="agrees"
    )
    # Question 3: text is too messy / missing math
    c3 = QuestionCorrection(
        question_number=3,
        needs_ocr_fallback=False,
        is_too_messy=True,
        question_type="descriptive_single"
    )

    questions_by_number = {
        1: {"question_number": 1, "stem": "In the circuit diagram shown, find current.", "options": {}},
        2: {"question_number": 2, "stem": "Unit of force is:", "options": {"A": "J", "B": "N"}, "marked_answer": "B"},
        3: {"question_number": 3, "stem": "Ny പൂ u _ v", "options": {}}
    }

    accepted, review_flags, ocr_fallbacks = _apply_post_processing_gate(
        [c1, c2, c3],
        questions_by_number,
        "dummy_job.pdf"
    )

    assert 1 in ocr_fallbacks, "Question 1 (needs_ocr_fallback) must be in ocr_fallbacks"
    assert 3 in ocr_fallbacks, "Question 3 (is_too_messy) must be in ocr_fallbacks"
    assert 2 not in ocr_fallbacks, "Question 2 (clean MCQ) must NOT be in ocr_fallbacks"
    print("✓ Post-processing OCR fallback gate test passed")

def test_prompt_rules():
    assert "DO NOT generate" in CORRECTION_SYSTEM_PROMPT or "DO NOT generate" in DIRECT_CORRECTION_SYSTEM_PROMPT
    assert "needs_ocr_fallback" in CORRECTION_SYSTEM_PROMPT or "PDF crop" in CORRECTION_SYSTEM_PROMPT
    print("✓ Prompt rules validation passed")

if __name__ == "__main__":
    test_pydantic_schema_and_defaults()
    test_ocr_fallback_gate()
    test_prompt_rules()
    print("\nALL LLM BEHAVIOR VERIFICATION TESTS PASSED SUCCESSFULLY!")
