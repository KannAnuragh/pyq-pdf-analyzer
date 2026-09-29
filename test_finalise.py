import sys
from unittest.mock import MagicMock
sys.modules['langchain_google_genai'] = MagicMock()
sys.modules['langchain_core'] = MagicMock()
sys.modules['langchain_core.prompts'] = MagicMock()
sys.modules['langchain_core.output_parsers'] = MagicMock()
sys.modules['langchain_core.runnables'] = MagicMock()
sys.modules['langchain_core.messages'] = MagicMock()
sys.modules['duckduckgo_search'] = MagicMock()
sys.modules['langgraph'] = MagicMock()
sys.modules['langgraph.graph'] = MagicMock()
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['google.genai.types'] = MagicMock()
sys.modules['pdf_qa_agent.prompts.templates'] = MagicMock()

import importlib
import pdf_qa_agent.ingest.block_classifier as bc

class DummySegment:
    def __init__(self, flat_text):
        self.flat_text = flat_text
        self.q_num = 1

segment = DummySegment("""A steady current of 8 mA flows through a wire. The number of electrons passing through a cross-section of the wire in 10 s is: [1 mark] 
(a) 4.0 \\times 10^{16} 
(b) 5.0 \\times 10^{17} 
(c) 1.6 \\times 10^{16} 
(d) 1.0 \\times 10^{17}""")

llm_result = {
    "stem": "A steady current of 8 mA flows through a wire. The number of electrons passing through a cross-section of the wire in 10 s is:",
    "options": {"D": "1.0 \\times 10^{17}"}
}

res = bc._finalise(llm_result, segment)
print("FINAL OPTIONS:", res["options"])
print("FINAL STEM:", res["stem"])
