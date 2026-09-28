import sys
sys.path.append("/home/anuragh/Trogon/qpaper scanner/pdf_qa_agent")
from ingest.math_normalizer import heal_student_readability

text = "For I:{r} $\\mu$ > 1, χ < 0, for II:{r} $\\mu$ < 1, χ > 0"
print("Before:", text)
print("After:", heal_student_readability(text))
