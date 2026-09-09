from graph.nodes import extract_node
from graph.state import GraphState

with open("raw_extracted.txt", "r") as f:
    text = f.read()

state = GraphState(
    raw_text=text,
    file_path="test",
    pages=[],
    extracted_blocks=[],
    quality_scores={},
    repair_log=[],
    vision_calls_count=0,
    processed_questions=[],
    summary={"total": 0, "validated": 0, "flagged": 0},
    token_usage={}
)

new_state = extract_node(state)
blocks = new_state.get("extracted_blocks", [])
print("Found", len(blocks), "blocks.")
for b in blocks[:5]:
    print("Q", b["question_number"])
