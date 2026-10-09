import asyncio
from pdf_qa_agent.graph.correction_node import fix_single_question_with_llm
import json

async def main():
    q_data = {
        "question_number": 1,
        "stem": "Answer the following questions:",
        "options": {
            "A": "Define force.",
            "B": "What is the SI unit of force?"
        },
        "marked_answer": None,
        "explanation": ""
    }
    res = await fix_single_question_with_llm(q_data)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
