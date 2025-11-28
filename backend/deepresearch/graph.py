from langgraph.graph import StateGraph, END, START
from .models import DeepResearchState
from .configuration import Config
from .nodes import (
    plan_research, 
    generate_queries, 
    perform_search, 
    write_report, 
    route_to_search,
    generate_feedback_queries,
    analyze_research_gaps
)

# Define the graph
workflow = StateGraph(DeepResearchState)

# Add nodes
workflow.add_node("plan_research", plan_research)
workflow.add_node("generate_queries", generate_queries)
workflow.add_node("perform_search", perform_search)
workflow.add_node("generate_feedback_queries", generate_feedback_queries)
workflow.add_node("write_report", write_report)
workflow.add_node("analyze_research_gaps", analyze_research_gaps)

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

# Logic to decide next step after search
def evaluate_progress(state: DeepResearchState):
    config = Config.from_env()
    feedback_mode = state.get("feedback_mode", config.feedback_mode)
    
    if feedback_mode == "auto":
        loop_count = state.get("feedback_loop_count", 0)
        if loop_count < config.max_feedback_loops:
            return "analyze_research_gaps"
        else:
            return "write_report"
    else:
        # Human mode: Go to review step (which interrupts)
        return "review_step"

workflow.add_conditional_edges(
    "perform_search",
    evaluate_progress,
    ["analyze_research_gaps", "write_report", "review_step"]
)

# Conditional logic from review step (Human Loop)
def check_human_feedback(state: DeepResearchState):
    if state.get("user_feedback"):
        return "generate_feedback_queries"
    return "write_report"

workflow.add_conditional_edges(
    "review_step",
    check_human_feedback,
    ["generate_feedback_queries", "write_report"]
)

# Conditional logic from auto feedback (Auto Loop)
def check_auto_feedback(state: DeepResearchState):
    if state.get("user_feedback"):
        return "generate_feedback_queries"
    return "write_report"

workflow.add_conditional_edges(
    "analyze_research_gaps",
    check_auto_feedback,
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
