import asyncio
from graph.correction_node import fix_single_question_with_llm
from dotenv import load_dotenv

load_dotenv()

async def test():
    q_data = {
        "question_number": 8,
        "stem": "An object of 3 cm height is placed at a distance of 60 cm from a convex mirror of focal length 30 cm. Find the (i) nature, (ii) position and (iii) size of the image formed.",
        "options": {},
        "marked_answer": None,
        "explanation": "(a) Here, height of object h = 3 cm\nu = -60cm, f = +30cm\nUsing the mirror formula, we have\n1 11\nvou f\n1.21\nv -60 30\n1 1 1 3\n> - = -4 + - ="
    }
    res = await fix_single_question_with_llm(q_data, provider="gemini")
    print(res)

if __name__ == "__main__":
    asyncio.run(test())
