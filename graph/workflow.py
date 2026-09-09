from langgraph.graph import StateGraph, START, END
from .state import GraphState
from .nodes import (
    extract_node, quality_score_node, repair_node,
    rule_check_node, solve_and_validate_node
)

def build_workflow():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("extract", extract_node)
    workflow.add_node("quality_score", quality_score_node)
    workflow.add_node("repair", repair_node)
    workflow.add_node("rule_check", rule_check_node)
    workflow.add_node("solve_and_validate", solve_and_validate_node)
    
    workflow.add_edge(START, "extract")
    workflow.add_edge("extract", "quality_score")
    workflow.add_edge("quality_score", "repair")
    workflow.add_edge("repair", "rule_check")
    workflow.add_edge("rule_check", "solve_and_validate")
    workflow.add_edge("solve_and_validate", END)
    
    return workflow.compile()

app_graph = build_workflow()
