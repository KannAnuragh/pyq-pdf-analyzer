import asyncio
from pdf_qa_agent.ingest.block_classifier import classify_block_with_llm
from pdf_qa_agent.ingest.layout_segmenter import LayoutElement, QuestionSegment
import json

class MockLLM:
    def invoke(self, messages):
        class Response:
            content = json.dumps({
                "stem": "1. If the frequency of an A.C is made 4 times of its initial value, the inductive reactance will be: [1 mark]",
                "options": {"A": "2 times", "B": "3 times", "C": "4 times", "D": "Unchanged"},
                "marked_answer": "A",
                "explanation": None,
                "answer_latex": None
            })
        return Response()

async def main():
    segment = QuestionSegment(
        elements=[
            LayoutElement(type="text", text="1. If the frequency of an A.C is made 4 times of its initial value, the inductive reactance will be: [1 mark]", y0=0, y1=10, bbox=[0,0,10,10], page=1, font_size=10.0, is_bold=False, x_indent=0.1),
            LayoutElement(type="text", text="(a) 2 times", y0=10, y1=20, bbox=[0,0,10,10], page=1, font_size=10.0, is_bold=False, x_indent=0.1),
            LayoutElement(type="text", text="(b) 3 times", y0=20, y1=30, bbox=[0,0,10,10], page=1, font_size=10.0, is_bold=False, x_indent=0.1),
            LayoutElement(type="text", text="(c) 4 times", y0=30, y1=40, bbox=[0,0,10,10], page=1, font_size=10.0, is_bold=False, x_indent=0.1),
            LayoutElement(type="text", text="(d) Unchanged", y0=40, y1=50, bbox=[0,0,10,10], page=1, font_size=10.0, is_bold=False, x_indent=0.1),
            LayoutElement(type="text", text="Answer:", y0=50, y1=60, bbox=[0,0,10,10], page=1, font_size=10.0, is_bold=True, x_indent=0.1),
            LayoutElement(type="text", text="(c) 4 times", y0=60, y1=70, bbox=[0,0,10,10], page=1, font_size=10.0, is_bold=False, x_indent=0.1),
        ],
        q_num=1,
        page_num=1,
        y0=0,
        y1=70
    )
    result = classify_block_with_llm(segment, llm=MockLLM())
    print("Result marked answer:", result.get("marked_answer"))

if __name__ == "__main__":
    asyncio.run(main())
