from langgraph.graph import StateGraph, START, END
from .state import GraphState
from .nodes import (
    extract_node, quality_score_node, repair_node,
    rule_check_node
)
from .correction_node import correction_and_validation_node

def build_workflow():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("extract", extract_node)
    workflow.add_node("quality_score", quality_score_node)
    workflow.add_node("repair", repair_node)
    workflow.add_node("correction_and_validation", correction_and_validation_node)
    workflow.add_node("rule_check", rule_check_node)
    
    workflow.add_edge(START, "extract")
    workflow.add_edge("extract", "quality_score")
    workflow.add_edge("quality_score", "repair")
    workflow.add_edge("repair", "correction_and_validation")
    workflow.add_edge("correction_and_validation", "rule_check")
    workflow.add_edge("rule_check", END)
    
    return workflow.compile()

app_graph = build_workflow()
