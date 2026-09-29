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
from graph.nodes import separate_stem_and_options
content = """A steady current of 8 mA flows through a wire. The number of electrons passing through a cross-section of the wire in 10 s is: [1 mark] (a) 4.0 \\times 10^{16} (b) 5.0 \\times 10^{17} (c) 1.6 \\times 10^{16} (d) 1.0 \\times 10^{17}"""
print(separate_stem_and_options(content))
