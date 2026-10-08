import os
import json
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from ingest.block_classifier import BLOCK_CLASSIFY_PROMPT, _clean_json

structured_text = """[BOLD] Assertion (A): Electric lines of force cross each other.
[BOLD] Reason (R): Electric field at this cross section point superimposes to give one electric field.
(a) If both Assertion and Reason are true and Reason is correct explanation of Assertion.
(b) If both Assertion and Reason are true but Reason is not the correct explanation of Assertion.
(c) If Assertion is true but Reason is false.
(d) If both Assertion and Reason are false.
Answer:
(d) If both Assertion and Reason are false.
"""

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
prompt = BLOCK_CLASSIFY_PROMPT.replace("{structured_text}", structured_text)
resp = llm.invoke([HumanMessage(content=prompt)])
print(resp.content)
