import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from pdf_qa_agent.ingest.block_classifier import BLOCK_CLASSIFY_PROMPT, _clean_json
import os
import json

async def main():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0.0
    )
    
    text = """1. If the frequency of an A.C is made 4 times of its initial value, the inductive reactance will be: [1 mark]
(a) 2 times
(b) 3 times
(c) 4 times
(d) Unchanged
Answer:
(c) 4 times"""

    prompt = BLOCK_CLASSIFY_PROMPT.replace("{structured_text}", text)
    response = llm.invoke(prompt)
    print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
