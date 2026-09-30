import asyncio
import os
import sys

from graph.correction_node import fix_single_question_with_llm
from dotenv import load_dotenv

load_dotenv()

async def test():
    q_data = {
        "question_number": 8,
        "stem": "A small piece of paper is stuck on the surface of the glass sphere of radius 10 cm and refractive index 1.5. The paper is viewed from the diametrically opposite side.",
        "options": {},
        "marked_answer": None,
        "explanation": "(a)\ni a.\nNy\nപു\nu\n_\nv\n(b) Using,\nMm നു സുന്ന്യ\nv വ R\nHere, u=—20cm,R= -10, മവു = 1.5,n,=1\nUpon substituting and calculating\n1-1\nv.40\n$v = -40cm$"
    }
    res = await fix_single_question_with_llm(q_data, provider="gemini")
    print("SUCCESS RESULT:", res.get("success"))
    print("NEEDS GEMINI OCR:", res.get("needs_gemini_ocr"))
    print(res.get("explanation"))

if __name__ == "__main__":
    asyncio.run(test())
