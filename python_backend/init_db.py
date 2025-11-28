import asyncio
from src.results_db import init_db
from src.configuration import Config
from rich.console import Console

console = Console()

async def main():
    config = Config.from_env()
    console.print(f"[bold blue]Initializing Database...[/bold blue]")
    console.print(f"Provider: [green]{config.db_provider}[/green]")
    console.print(f"URI: [dim]{config.db_uri}[/dim]")
    
    try:
        await init_db()
        console.print("[bold green]Success! Database initialized.[/bold green]")
        
        if config.db_provider == "postgres":
            # For Postgres, we also need to ensure LangGraph tables exist
            # This is usually handled by checkpointer.setup() in main.py, 
            # but we can try to trigger it here if needed.
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            async with AsyncPostgresSaver.from_conn_string(config.db_uri) as checkpointer:
                await checkpointer.setup()
            console.print("[bold green]LangGraph tables verified.[/bold green]")
            
    except Exception as e:
        console.print(f"[bold red]Error initializing database:[/bold red] {e}")

if __name__ == "__main__":
    asyncio.run(main())
