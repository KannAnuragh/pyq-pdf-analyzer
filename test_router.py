import sys, asyncio
sys.path.insert(0, '/home/anuragh/Trogon/qpaper scanner/pdf_qa_agent')
from ingest.router import parse_pdf

async def main():
    questions = parse_pdf('data/uploads/ed3a3b72-550a-4a1a-a3ac-8ed35fe422da_12CB CH-4 Moving charges & Magnetism (DPP).pdf')
    for q in questions:
        print(f"Q{q['question_number']}: {q['question_text'][:100]!r}...")

asyncio.run(main())
