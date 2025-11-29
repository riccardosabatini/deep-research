import asyncio
import os
import uuid
from typing import Any
from dotenv import load_dotenv
import aiosqlite

# Load environment variables (API keys, etc.)
load_dotenv()

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from deepresearch.graph import workflow
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from deepresearch.results_db import init_db
from rich.logging import RichHandler
import logging
from deepresearch.configuration import Config
from deepresearch.models import DeepResearchState
from langgraph.types import Send
from pydantic import BaseModel
import json
import argparse
from langchain_community.cache import RedisCache
from langchain_core.globals import set_llm_cache
from redis import Redis

# Custom JSON serialization for LangGraph
def custom_encoder(obj):
    if isinstance(obj, Send):
        return {"__type__": "Send", "node": obj.node, "arg": obj.arg}
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def custom_decoder(obj):
    if "__type__" in obj and obj["__type__"] == "Send":
        return Send(obj["node"], obj["arg"])
    return obj

def json_dumps(obj):
    return json.dumps(obj, default=custom_encoder)

def json_loads(obj):
    return json.loads(obj, object_hook=custom_decoder)

class JsonSerializer:
    def dumps(self, obj) -> bytes:
        # console.print(f"[DEBUG] Serializing object type: {type(obj)}")
        try:
            return json.dumps(obj, default=custom_encoder).encode("utf-8")
        except TypeError as e:
            console.print(f"[ERROR] Serialization failed for: {obj}")
            raise e

    def loads(self, data: bytes) -> Any:
        return json.loads(data.decode("utf-8"), object_hook=custom_decoder)

    def dumps_typed(self, obj) -> tuple[str, bytes]:
        # console.print(f"[DEBUG] dumps_typed called for {type(obj)}")
        if isinstance(obj, bytes):
            return "bytes", obj
        return "json", self.dumps(obj)

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        type_, data_ = data
        if type_ == "bytes":
            return data_
        return self.loads(data_)

# Setup logging
logging.basicConfig(
    level=Config.from_env().log_level,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

console = Console()

async def run_research(query: str, thread_id: str = None, feedback_mode: str = "human", report_pages: int = 5):
    """
    Runs the deep research process with persistence.
    """
    console.print(f"[bold green]Starting research session...[/bold green]")
    
    # Initialize configuration
    app_config = Config.from_env()
    
    # Setup LLM Cache
    if app_config.redis_enabled:
        console.print(f"[bold blue]Enabling Redis Cache at {app_config.redis_url}[/bold blue]")
        set_llm_cache(RedisCache(redis_=Redis.from_url(app_config.redis_url)))
    
    # Initialize DB (for search results)
    await init_db()

    # Setup Checkpointer
    if app_config.db_provider == "postgres":
        async with AsyncPostgresSaver.from_conn_string(
            app_config.db_uri,
            serde=JsonSerializer()
        ) as checkpointer:
            await checkpointer.setup()
            return await _run_graph(checkpointer, query, thread_id, feedback_mode, report_pages)
    else:
        # Default to SQLite
        async with aiosqlite.connect(app_config.db_uri) as conn:
            checkpointer = AsyncSqliteSaver(conn, serde=JsonSerializer())
            await checkpointer.setup()
            return await _run_graph(checkpointer, query, thread_id, feedback_mode, report_pages)

async def _run_graph(checkpointer, query: str, thread_id: str, feedback_mode: str, report_pages: int):
    # Create the graph with the checkpointer
    app = workflow.compile(
        checkpointer=checkpointer, 
        interrupt_before=["review_step"]
    )
    
    # Generate a thread ID if not provided
    if not thread_id:
        thread_id = str(uuid.uuid4())
    run_config = {"configurable": {"thread_id": thread_id}}
    
    console.print(Panel(f"Research Thread ID: {thread_id}", title="[bold blue]Deep Research[/bold blue]"))
    
    # Check if state exists
    state = await app.aget_state(run_config)
    
    if not state.values:
        # New research session
        if not query:
            raise ValueError("Query is required for a new research session.")
            
        console.print(f"[bold green]Starting new research session...[/bold green]")
        initial_state = {
            "query": query,
            "report_plan": "",
            "serp_queries": [],
            "search_results": [],
            "user_feedback": None,
            "feedback_loop_count": 0,
            "feedback_mode": feedback_mode,
            "report_pages": report_pages,
            "final_report": ""
        }
        
        # Run until the interrupt
        async for event in app.astream(initial_state, config=run_config):
            for key, value in event.items():
                console.print(f"--- Step Completed: [bold cyan]{key}[/bold cyan] ---")
    else:
        console.print(f"[bold yellow]Resuming existing research session...[/bold yellow]")

    # 2. Review Loop
    while True:
        # Inspect State at Interrupt
        snapshot = await app.aget_state(run_config)
        if not snapshot.values:
            break 
            
        # Check if we are done (final_report exists and next is empty?)
        if "final_report" in snapshot.values and snapshot.values["final_report"]:
             pass

        current_results = snapshot.values.get("search_results", [])
        console.print(f"\n[bold yellow]=== Search Results ({len(current_results)} items) ===[/bold yellow]")
        for i, res in enumerate(current_results):
            # res is now a dict
            console.print(f"\n[bold]Result {i+1}:[/bold] {res['query']}")
            console.print(f"[italic]Goal:[/italic] {res['research_goal']}")
            console.print(f"Learnings: {len(res['learnings'])} blocks")
            console.print(f"Sources: {len(res['sources'])}")
            
        # 3. Human-in-the-Loop: Ask for feedback
        # Only ask for feedback if we are in human mode AND at the review_step
        # If we are in auto mode, the graph shouldn't have interrupted unless it finished or hit a limit?
        # Actually, interrupt_before=["review_step"] is set globally.
        # But in Auto mode, we don't go to review_step unless we want to stop?
        # Wait, the graph logic says:
        # if auto: analyze_research_gaps -> ...
        # if human: review_step
        
        # So if we are in auto mode, we should NOT be hitting this loop unless the graph finished.
        # BUT, if we are resuming, we might be in a state.
        
        # If the graph finished, astream returns.
        # If the graph interrupted, we are here.
        
        # If we are in auto mode, we shouldn't have interrupted at review_step because we don't go there.
        # So if we are here, it means we are either done or in human mode.
        
        # Let's check if there are tasks to run.
        if not snapshot.next:
             console.print("[bold green]Research Completed.[/bold green]")
             break
             
        console.print(Panel("[REVIEW] Do you want to add more queries based on these results?\nType your feedback/suggestion to generate new queries, or press Enter to finish.", title="[bold red]User Feedback[/bold red]"))
        
        feedback = input("> ")
        
        if not feedback.strip():
            # No feedback, proceed to report
            console.print("\n[bold green]--- Proceeding to Final Report ---[/bold green]")
            # We update state to ensure user_feedback is None (it should be None by default, but to be safe)
            # await app.aupdate_state(run_config, {"user_feedback": None})
            
            # Resume execution (runs review_step -> write_report)
            async for event in app.astream(None, config=run_config):
                for key, value in event.items():
                    console.print(f"--- Step Completed: [bold cyan]{key}[/bold cyan] ---")
            break
        else:
            # User provided feedback
            console.print(f"\n[bold]Generating new queries based on:[/bold] '{feedback}'")
            # Update state with feedback
            await app.aupdate_state(run_config, {"user_feedback": feedback}, as_node="review_step")
            
            # Resume execution (runs review_step -> generate_feedback_queries -> perform_search -> Pauses at Review)
            async for event in app.astream(None, config=run_config):
                for key, value in event.items():
                    console.print(f"--- Step Completed: [bold cyan]{key}[/bold cyan] ---")
            
    # Get final state
    final_state = await app.aget_state(run_config)
    return final_state.values.get("final_report")

def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description="Deep Research Assistant")
    parser.add_argument("query", nargs="?", help="The research query (optional if resuming)")
    parser.add_argument("--thread-id", help="Thread ID to resume an existing research session")
    parser.add_argument("--feedback-mode", choices=["human", "auto"], default="human", help="Feedback mode: 'human' for manual review, 'auto' for LLM review")
    parser.add_argument("--report-pages", type=int, default=5, help="Target number of pages for the final report")
    return parser.parse_args(args)

if __name__ == "__main__":
    args = parse_arguments()
    
    if not args.query and not args.thread_id:
        parser.print_help()
        sys.exit(1)
        
    # Check if input is a file
    query = args.query
    if query and os.path.isfile(query):
        console.print(f"[dim]Reading query from file: {query}[/dim]")
        with open(query, "r") as f:
            query = f.read().strip()
    
    try:
        final_report = asyncio.run(run_research(query=query, thread_id=args.thread_id, feedback_mode=args.feedback_mode, report_pages=args.report_pages))
        if final_report:
            console.print(Panel(Markdown(final_report), title="[bold green]Final Report[/bold green]"))
        else:
            console.print("[bold red]Research finished without generating a report.[/bold red]")
    except Exception as e:
        console.print(f"[bold red]An error occurred:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())
