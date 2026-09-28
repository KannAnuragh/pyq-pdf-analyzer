import sys
from pdf_qa_agent.graph.nodes import parse_questions_from_text, separate_stem_and_options
from pdf_qa_agent.ingest.option_answer_separator import classify_question_type, clean_question_options_and_answer

def run_tests():
    print("=== Testing MCQ Options with Equations ===")

    # Test 1: Physics formulas in options (e.g. \frac{\mu_0 I}{2R})
    q1 = {
        "question_number": 1,
        "stem": "The magnetic field at the centre of circular loop of radius R carrying current I is:",
        "options": {
            "A": r"\frac{\mu_0 I}{2R}",
            "B": r"\frac{\mu_0 I}{4\pi R}",
            "C": r"\frac{\mu_0 I}{2R}(1-\frac{1}{\pi})",
            "D": "zero"
        },
        "marked_answer": "A",
        "explanation": "At the centre of the loop, B = \\frac{\\mu_0 I}{2R}."
    }
    clean_question_options_and_answer(q1)
    assert q1["question_type"] == "mcq", f"Test 1 failed: Expected mcq, got {q1['question_type']}"
    assert len(q1["options"]) == 4, f"Test 1 failed: Expected 4 options, got {len(q1['options'])}"
    print("✓ Test 1: Physics formulas in options cleanly retained and classified as MCQ.")

    # Test 2: Mathematical equations (y = mx + c)
    q2 = {
        "question_number": 2,
        "stem": "The equation of a straight line passing through origin is:",
        "options": {
            "A": "$y = mx$",
            "B": "$y = mx + c$",
            "C": "$y = x^2$",
            "D": "$y = \\sqrt{x}$"
        },
        "explanation": "For a line passing through the origin, the intercept c = 0."
    }
    clean_question_options_and_answer(q2)
    assert q2["question_type"] == "mcq", f"Test 2 failed: Expected mcq, got {q2['question_type']}"
    assert len(q2["options"]) == 4, f"Test 2 failed: Expected 4 options, got {len(q2['options'])}"
    print("✓ Test 2: Pure algebraic equations in options classified as MCQ.")

    # Test 3: Scientific notation with units
    q3 = {
        "question_number": 3,
        "stem": "A current of 10 A flows in wire. The magnetic field at 10 cm is:",
        "options": {
            "A": "$2 \\times 10^{-5} \\, \\text{T}$, acting downwards",
            "B": "$2 \\times 10^{-5} \\, \\text{T}$, acting upwards",
            "C": "$4 \\times 10^{-5} \\, \\text{T}$, acting downwards",
            "D": "$4 \\times 10^{-5} \\, \\text{T}$, acting upwards"
        },
        "marked_answer": "A"
    }
    clean_question_options_and_answer(q3)
    assert q3["question_type"] == "mcq", f"Test 3 failed: Expected mcq, got {q3['question_type']}"
    assert len(q3["options"]) == 4, f"Test 3 failed: Expected 4 options, got {len(q3['options'])}"
    print("✓ Test 3: Scientific units and directional formulas retained as MCQ.")

    # Test 4: Ratios with Greek symbols
    q4 = {
        "question_number": 4,
        "stem": "The ratio of magnetic moments is:",
        "options": {
            "A": "2 ∶ 𝜋",
            "B": "𝜋 ∶ 2",
            "C": "𝜋 ∶ 4",
            "D": "4 ∶ 𝜋"
        },
        "marked_answer": "C",
        "explanation": "\\pi : 4"
    }
    clean_question_options_and_answer(q4)
    assert q4["question_type"] == "mcq", f"Test 4 failed: Expected mcq, got {q4['question_type']}"
    assert len(q4["options"]) == 4, f"Test 4 failed: Expected 4 options, got {len(q4['options'])}"
    print("✓ Test 4: Math ratios with Greek symbols retained as MCQ.")

    # Test 5: Options with equations when explanation has step-by-step numbered steps
    q5 = {
        "question_number": 5,
        "stem": "Find the equivalent capacitance:",
        "options": {
            "A": "$2\\,\\mu\\text{F}$",
            "B": "$4\\,\\mu\\text{F}$",
            "C": "$6\\,\\mu\\text{F}$",
            "D": "$8\\,\\mu\\text{F}$"
        },
        "explanation": "Step 1. (a) is in series with (b). Step 2. Calculate parallel combination to get (b)."
    }
    clean_question_options_and_answer(q5)
    assert q5["question_type"] == "mcq", f"Test 5 failed: Expected mcq, got {q5['question_type']}"
    assert len(q5["options"]) == 4, f"Test 5 failed: Expected 4 options, got {len(q5['options'])}"
    print("✓ Test 5: Options with equations when explanation contains step numbers retained as MCQ.")

    # Test 6: Raw text parsing with equation options
    raw_doc = r"""1. The relation between velocity v and time t is:
(a) $v = u + at$
(b) $v = u - at$
(c) $v^2 = u^2 + 2as$
(d) $s = ut + \frac{1}{2}at^2$
Answer: (a) $v = u + at$"""

    stem6, opts6, ans6, exp6 = separate_stem_and_options(raw_doc)
    assert len(opts6) == 4, f"Test 6 failed: Expected 4 options, got {len(opts6)}: {opts6}"
    assert ans6 == "A", f"Test 6 failed: Expected marked answer A, got {ans6}"
    print("✓ Test 6: Raw document separation extracts all 4 equation options seamlessly.")

    print("\n=== ALL MCQ EQUATION TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_tests()
