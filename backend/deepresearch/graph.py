from langgraph.graph import StateGraph, END, START
from .models import DeepResearchState
from .nodes import (
    plan_research, 
    generate_queries, 
    perform_search, 
    write_report, 
    route_to_search,
    generate_feedback_queries
)

# Define the graph
workflow = StateGraph(DeepResearchState)

# Add nodes
workflow.add_node("plan_research", plan_research)
workflow.add_node("generate_queries", generate_queries)
workflow.add_node("perform_search", perform_search)
workflow.add_node("generate_feedback_queries", generate_feedback_queries)
workflow.add_node("write_report", write_report)

# Dummy node for the review step (pass-through)
def review_step(state: DeepResearchState):
    return state
workflow.add_node("review_step", review_step)

# Add edges
workflow.add_edge(START, "plan_research")
workflow.add_edge("plan_research", "generate_queries")

# Conditional edge for parallel execution (Map step)
workflow.add_conditional_edges(
    "generate_queries", 
    route_to_search,
    ["perform_search"]
)

# After search, go to review step
workflow.add_edge("perform_search", "review_step")

# Conditional logic from review step
def check_feedback(state: DeepResearchState):
    if state.get("user_feedback"):
        return "generate_feedback_queries"
    return "write_report"

workflow.add_conditional_edges(
    "review_step",
    check_feedback,
    ["generate_feedback_queries", "write_report"]
)

# From feedback generation, route back to search
workflow.add_conditional_edges(
    "generate_feedback_queries",
    route_to_search,
    ["perform_search"]
)

workflow.add_edge("write_report", END)

# Compile the graph
graph = workflow.compile()
