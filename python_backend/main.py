import asyncio
import os
import uuid
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.graph import workflow

# Load environment variables (API keys, etc.)
load_dotenv()

async def run_research(query: str, thread_id: str = None):
    """
    Runs the deep research process with persistence.
    """
    if not thread_id:
        thread_id = str(uuid.uuid4())
        
    print(f"Starting research session: {thread_id}")
    
    # Use SQLite for persistence    # Use SQLite for persistence
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        # Compile with an interrupt before the review step
        graph = workflow.compile(
            checkpointer=checkpointer, 
            interrupt_before=["review_step"]
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # 1. Start the process (runs Plan -> Generate -> Search -> Pauses at Review)
        print(f"--- Starting Research (Thread: {thread_id}) ---")
        initial_state = {
            "query": query,
            "report_plan": "",
            "serp_queries": [],
            "search_results": [],
            "user_feedback": None,
            "final_report": ""
        }
        
        # Run until the interrupt
        async for event in graph.astream(initial_state, config=config):
            for key, value in event.items():
                print(f"\n--- Step Completed: {key} ---")
        
        # 2. Review Loop
        while True:
            # Inspect State at Interrupt
            snapshot = await graph.aget_state(config)
            if not snapshot.values:
                break # Should not happen if paused
                
            current_results = snapshot.values.get("search_results", [])
            
            print(f"\n=== Search Results ({len(current_results)} items) ===")
            for i, res in enumerate(current_results):
                print(f"\n--- Result {i+1}: {res.query} ---")
                print(f"Goal: {res.research_goal}")
                print(f"Learnings: {len(res.learnings)} blocks")
                print(f"Sources: {len(res.sources)}")
                
            # 3. Human-in-the-Loop: Ask for feedback
            print("\n[REVIEW] Do you want to add more queries based on these results?")
            print("Type your feedback/suggestion to generate new queries, or press Enter to finish.")
            
            feedback = input("> ")
            
            if not feedback.strip():
                # No feedback, proceed to report
                print("\n--- Proceeding to Final Report ---")
                # We update state to ensure user_feedback is None (it should be None by default, but to be safe)
                await graph.aupdate_state(config, {"user_feedback": None})
                
                # Resume execution (runs review_step -> write_report)
                async for event in graph.astream(None, config=config):
                    for key, value in event.items():
                        print(f"\n--- Step Completed: {key} ---")
                break
            else:
                # User provided feedback
                print(f"\n--- Generating new queries based on: '{feedback}' ---")
                # Update state with feedback
                await graph.aupdate_state(config, {"user_feedback": feedback})
                
                # Resume execution (runs review_step -> generate_feedback_queries -> perform_search -> Pauses at Review)
                async for event in graph.astream(None, config=config):
                    for key, value in event.items():
                        print(f"\n--- Step Completed: {key} ---")
                
                # Loop continues, showing new results...
                
        # Get final state
        final_state = await graph.aget_state(config)
        return final_state.values.get("final_report")

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <query>")
        sys.exit(1)
        
    query = sys.argv[1]
    
    try:
        final_report = asyncio.run(run_research(query))
        print("\n=== FINAL REPORT ===\n")
        print(final_report)
    except Exception as e:
        print(f"An error occurred: {e}")
