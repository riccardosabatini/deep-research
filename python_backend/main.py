import asyncio
import os
import uuid
from typing import Any
from dotenv import load_dotenv

# Load environment variables (API keys, etc.)
load_dotenv()

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from src.graph import workflow
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from src.results_db import init_db
from rich.logging import RichHandler
import logging
from src.configuration import Config
from src.models import DeepResearchState
from langgraph.types import Send
from pydantic import BaseModel
import json
import argparse

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
        # Postgres saver might expect bytes or string depending on implementation
        # Usually jsonb expects string, but some serializers return bytes.
        # Let's return string as json.dumps returns string.
        # Wait, SerializerProtocol usually expects bytes?
        # Let's check if it fails. If it expects bytes, I'll encode.
        return json.dumps(obj, default=custom_encoder).encode("utf-8")

    def loads(self, data: bytes) -> Any:
        return json.loads(data.decode("utf-8"), object_hook=custom_decoder)

# Setup logging
logging.basicConfig(
    level=Config.from_env().log_level,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

console = Console()

async def run_research(query: str, thread_id: str = None):
    """
    Runs the deep research process with persistence.
    """
    console.print(f"[bold green]Starting research session...[/bold green]")
    
    # Initialize configuration
    app_config = Config.from_env()
    
    # Setup Checkpointer
    if app_config.db_provider == "postgres":
        checkpointer_cm = AsyncPostgresSaver.from_conn_string(
            app_config.db_uri,
            serde=JsonSerializer()
        )
    else:
        # Default to SQLite
        checkpointer_cm = AsyncSqliteSaver.from_conn_string(app_config.db_uri)

    # Initialize DB (for search results)
    await init_db()

    async with checkpointer_cm as checkpointer:
        # For Postgres, we need to setup the tables
        if app_config.db_provider == "postgres":
            await checkpointer.setup()
            
        # Create the graph with the checkpointer
        app = workflow.compile(
            checkpointer=checkpointer, 
            interrupt_before=["review_step"]
        )
        
        # Generate a thread ID if not provided
        if not thread_id:
            thread_id = str(uuid.uuid4())
        run_config = {"configurable": {"thread_id": thread_id}}
        
        # 1. Start the process (runs Plan -> Generate -> Search -> Pauses at Review)
        console.print(Panel(f"Starting Research (Thread: {thread_id})", title="[bold blue]Deep Research[/bold blue]"))
        initial_state = {
            "query": query,
            "report_plan": "",
            "serp_queries": [],
            "search_results": [],
            "user_feedback": None,
            "final_report": ""
        }
        
        # Run until the interrupt
        async for event in app.astream(initial_state, config=run_config):
            for key, value in event.items():
                console.print(f"--- Step Completed: [bold cyan]{key}[/bold cyan] ---")
        
        # 2. Review Loop
        while True:
            # Inspect State at Interrupt
            snapshot = await app.aget_state(run_config)
            if not snapshot.values:
                break # Should not happen if paused
                
            current_results = snapshot.values.get("search_results", [])
            
            console.print(f"\n[bold yellow]=== Search Results ({len(current_results)} items) ===[/bold yellow]")
            for i, res in enumerate(current_results):
                console.print(f"\n[bold]Result {i+1}:[/bold] {res.query}")
                console.print(f"[italic]Goal:[/italic] {res.research_goal}")
                console.print(f"Learnings: {len(res.learnings)} blocks")
                console.print(f"Sources: {len(res.sources)}")
                
            # 3. Human-in-the-Loop: Ask for feedback
            console.print(Panel("[REVIEW] Do you want to add more queries based on these results?\nType your feedback/suggestion to generate new queries, or press Enter to finish.", title="[bold red]User Feedback[/bold red]"))
            
            feedback = input("> ")
            
            if not feedback.strip():
                # No feedback, proceed to report
                console.print("\n[bold green]--- Proceeding to Final Report ---[/bold green]")
                # We update state to ensure user_feedback is None (it should be None by default, but to be safe)
                await app.aupdate_state(run_config, {"user_feedback": None})
                
                # Resume execution (runs review_step -> write_report)
                async for event in app.astream(None, config=run_config):
                    for key, value in event.items():
                        console.print(f"--- Step Completed: [bold cyan]{key}[/bold cyan] ---")
                break
            else:
                # User provided feedback
                console.print(f"\n[bold]Generating new queries based on:[/bold] '{feedback}'")
                # Update state with feedback
                await app.aupdate_state(run_config, {"user_feedback": feedback})
                
                # Resume execution (runs review_step -> generate_feedback_queries -> perform_search -> Pauses at Review)
                async for event in app.astream(None, config=run_config):
                    for key, value in event.items():
                        console.print(f"--- Step Completed: [bold cyan]{key}[/bold cyan] ---")
                
                # Loop continues, showing new results...
                
        # Get final state
        final_state = await app.aget_state(run_config)
        return final_state.values.get("final_report")

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        console.print("[bold red]Usage:[/bold red] python main.py <query_or_file_path>")
        sys.exit(1)
        
    input_arg = sys.argv[1]
    
    # Check if input is a file
    if os.path.isfile(input_arg):
        console.print(f"[dim]Reading query from file: {input_arg}[/dim]")
        with open(input_arg, "r") as f:
            query = f.read().strip()
    else:
        query = input_arg
    
    try:
        final_report = asyncio.run(run_research(query))
        console.print(Panel(Markdown(final_report), title="[bold green]Final Report[/bold green]"))
    except Exception as e:
        console.print(f"[bold red]An error occurred:[/bold red] {e}")
