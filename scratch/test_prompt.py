import asyncio
import json
from graph.nodes import _clean_json_output
from prompts.templates import SOLVE_PROMPT
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

prompt = SOLVE_PROMPT.format(
    question_type="descriptive_subquestions",
    stem="An object of 3 cm height is placed at a distance of 60 cm from a convex mirror of focal length 30 cm. Find the (i) nature, (ii) position and (iii) size of the image formed.",
    sub_parts="{}",
    options="{}",
    explanation="(a) Here, height of object h = 3 cm\nu = -60cm, f = +30cm\nUsing the mirror formula, we have\n1 11\nvou f\n1.21\nv -60 30\n1 1 1 3\n> - = -4 + - ="
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
resp = llm.invoke([HumanMessage(content=prompt)])
print(resp.content)
