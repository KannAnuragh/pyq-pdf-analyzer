import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from pdf_qa_agent.ingest.text_cleaner import clean_indic_text, auto_fix_deterministic
    from pdf_qa_agent.ingest.router import clean_latex_equation
except ImportError:
    from ingest.text_cleaner import clean_indic_text, auto_fix_deterministic
    from ingest.router import clean_latex_equation

def test_samples():
    cases = [
        # Case 1: Raw text with unicode multiplication cross '×' and superscripts
        (
            "96 = 2⁵ × 3, 404 = 2² × 101.",
            "അഭാജ്യ ഘടകങ്ങളായി എഴുതുമ്പോൾ:\n96 = 2⁵ × 3, 404 = 2² × 101."
        ),
        # Case 2: Text already corrupted into x^3 or x 101
        (
            "96 = 2^5 x^3 404 = 2^2 x 101",
            "96 = 2^5 x^3 404 = 2^2 x 101"
        ),
        # Case 3: Standard LaTeX \times
        (
            "$96 = 2^5 \\times 3, 404 = 2^2 \\times 101$",
            "$96 = 2^5 \\times 3, 404 = 2^2 \\times 101$"
        ),
        # Case 4: Text cleaner on explanation
        (
            "അഭാജ്യ ഘടകങ്ങളായി എഴുതുമ്പോൾ 96 = 2⁵ × 3, 404 = 2² × 101.",
            "അഭാജ്യ ഘടകങ്ങളായി എഴുതുമ്പോൾ 96 = 2⁵ × 3, 404 = 2² × 101."
        )
    ]

    print("=== Testing clean_indic_text ===")
    for raw, desc in cases:
        cleaned, _ = clean_indic_text(raw)
        print(f"\n[Raw]:     {raw}")
        print(f"[Cleaned]: {cleaned}")
        assert "x^3" not in cleaned, f"Failed: x^3 still in cleaned text: {cleaned}"
        assert "\\times" in cleaned or "×" in cleaned, f"Failed: multiplication symbol lost: {cleaned}"

    print("\n=== Testing clean_latex_equation ===")
    latex_cases = [
        "96 = 2^5 \\times 3, 404 = 2^2 \\times 101",
        "96 = 2^5 x^3 404 = 2^2 x 101",
        "2^5 \\times 3"
    ]
    for lc in latex_cases:
        res = clean_latex_equation(lc)
        print(f"\n[Input]:  {lc}")
        print(f"[Output]: {res}")
        assert "x^3" not in res, f"Failed: x^3 in {res}"
        assert "\\times" in res, f"Failed: \\times missing in {res}"

    print("\nALL BACKEND TESTS PASSED!")

if __name__ == "__main__":
    test_samples()
