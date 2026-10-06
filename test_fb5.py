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
from graph.nodes import separate_stem_and_options
content = """1. A steady current of 8 mA flows through a wire. The number of electrons passing 
through a cross-section of the wire in 10 s is: 
 
[1 mark] 
(a) 4.0 × 1016 
(b) 5.0 × 1017 
(c) 1.6 × 1016 
(d) 1.0 × 1017 
Answer: 
(b) 5.0 × 1017"""
print(separate_stem_and_options(content))
