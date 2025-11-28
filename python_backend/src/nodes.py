import asyncio
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.config import RunnableConfig
from langgraph.types import Send

from rich.console import Console
from rich.panel import Panel

from .models import (
    DeepResearchState, 
    DeepResearchSearchTask, 
    DeepResearchQueryList,
    DeepResearchSearchResult, 
    SearchResultItem,
    ImageSource
)
from .prompts import (
    report_plan_prompt, 
    serp_queries_prompt, 
    process_search_result_prompt, 
    final_report_prompt,
    review_prompt
)
from .tools import SearchTools
from .configuration import Config
from .results_db import get_search_result, save_search_result

# Initialize
search_tools = SearchTools()
console = Console()

def get_llm(config: RunnableConfig, model_type: str = "task"):
    # ... (keep existing get_llm implementation) ...
    """
    Helper to get the LLM instance based on configuration.
    """
    # Load config from configurable or env
    configurable = config.get("configurable", {}) if config else {}
    env_config = Config.from_env()
    
    # Check if we passed a full Config object or just individual keys
    if model_type == "thinking":
        model_name = configurable.get("thinking_model", env_config.thinking_model)
    else:
        model_name = configurable.get("task_model", env_config.task_model)
        
    base_url = configurable.get("base_url", env_config.base_url)
    api_key = configurable.get("api_key", env_config.api_key)
    provider = configurable.get("provider", env_config.provider).lower()
    
    if provider == "anthropic":
        return ChatAnthropic(
            model_name=model_name,
            temperature=0,
            api_key=api_key if api_key else None,
            base_url=base_url if base_url else None
        )
    elif provider == "google" or provider == "google_genai":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            google_api_key=api_key if api_key else None
        )
    elif provider == "groq":
        return ChatGroq(
            model_name=model_name,
            temperature=0,
            api_key=api_key if api_key else None
        )
    
    # Default to OpenAI (works for DeepSeek, OpenRouter, etc. via base_url)
    return ChatOpenAI(
        model=model_name, 
        temperature=0,
        base_url=base_url if base_url else None,
        api_key=api_key if api_key else None
    )

async def plan_research(state: DeepResearchState, config: RunnableConfig):
    """
    Node: Generates the research plan based on the user query.
    Uses: Thinking Model
    """
    console.print(Panel(f"Planning Research for: {state['query']}", title="[bold blue]Plan Research[/bold blue]"))
    llm = get_llm(config, "thinking")
    chain = report_plan_prompt | llm | StrOutputParser()
    plan = await chain.ainvoke({"query": state["query"]})
    return {"report_plan": plan}

async def generate_queries(state: DeepResearchState, config: RunnableConfig):
    """
    Node: Generates SERP queries based on the research plan.
    Uses: Thinking Model
    """
    console.print(Panel("Generating Queries...", title="[bold blue]Generate Queries[/bold blue]"))
    llm = get_llm(config, "thinking")
    
    # Use the wrapper model
    structured_llm = llm.with_structured_output(DeepResearchQueryList)
    chain = serp_queries_prompt | structured_llm
    
    result = await chain.ainvoke({"plan": state["report_plan"]})
    return {"serp_queries": result.queries}

async def perform_search(task: DeepResearchSearchTask, config: RunnableConfig):
    """
    Node: Executes a single search task and processes the results.
    Uses: Task Model (for processing results)
    """
    # Check cache first
    cached = await get_search_result(task.query)
    if cached:
        console.print(f"[dim]Found cached result for: {task.query}[/dim]")
        sources = [SearchResultItem(**s) for s in cached["raw_result"]["sources"]]
        images = [ImageSource(**i) for i in cached["raw_result"]["images"]]
        learnings = cached["learnings"]
        
        result = DeepResearchSearchResult(
            query=task.query,
            research_goal=task.research_goal,
            learnings=[learnings],
            sources=sources,
            images=images
        )
        return {"search_results": [result]}

    console.print(f"[green]Executing Search:[/green] {task.query}")
    llm = get_llm(config, "task")
    
    # 1. Execute Search
    search_data = await search_tools.perform_search(task.query)
    sources: List[SearchResultItem] = search_data["sources"]
    images: List[ImageSource] = search_data["images"]
    
    # 2. Process Results (Synthesize Learnings)
    # Format context for the prompt
    context_str = "\n\n".join([
        f"<content index=\"{i+1}\" url=\"{s.url}\">\n{s.content}\n</content>"
        for i, s in enumerate(sources)
    ])
    
    chain = process_search_result_prompt | llm | StrOutputParser()
    learnings = await chain.ainvoke({
        "query": task.query,
        "researchGoal": task.research_goal,
        "context": context_str
    })
    
    # Save to DB
    await save_search_result(
        query=task.query,
        raw_result={"sources": [s.dict() for s in sources], "images": [i.dict() for i in images]},
        learnings=learnings
    )
    
    # 3. Return Result
    result = DeepResearchSearchResult(
        query=task.query,
        research_goal=task.research_goal,
        learnings=[learnings],
        sources=sources,
        images=images
    )
    
    return {"search_results": [result]}

async def generate_feedback_queries(state: DeepResearchState, config: RunnableConfig):
    """
    Node: Generates new queries based on user feedback.
    Uses: Thinking Model
    """
    console.print(Panel("Generating Feedback Queries...", title="[bold blue]Feedback Loop[/bold blue]"))
    llm = get_llm(config, "thinking")
    
    # Aggregate learnings for context
    all_learnings = []
    for result in state["search_results"]:
        all_learnings.extend(result.learnings)
    learnings_str = "\n".join(all_learnings)
    
    structured_llm = llm.with_structured_output(DeepResearchQueryList)
    chain = review_prompt | structured_llm
    
    result = await chain.ainvoke({
        "plan": state["report_plan"],
        "learnings": learnings_str,
        "suggestion": state["user_feedback"]
    })
    
    return {
        "serp_queries": result.queries, 
        "user_feedback": None 
    }

async def write_report(state: DeepResearchState, config: RunnableConfig):
    """
    Node: Synthesizes the final report.
    Uses: Task Model
    """
    console.print(Panel("Writing Final Report...", title="[bold blue]Write Report[/bold blue]"))
    llm = get_llm(config, "task")
    
    # Aggregate learnings, sources, and images
    all_learnings = []
    all_sources = []
    all_images = []
    
    for result in state["search_results"]:
        all_learnings.extend(result.learnings)
        all_sources.extend(result.sources)
        all_images.extend(result.images)
        
    # Format for prompt
    learnings_str = "\n".join(all_learnings)
    
    sources_str = "\n".join([
        f"<source index=\"{i+1}\" url=\"{s.url}\">\n{s.title}\n</source>"
        for i, s in enumerate(all_sources)
    ])
    
    images_str = "\n".join([
        f"{i+1}. ![{img.description}]({img.url})"
        for i, img in enumerate(all_images)
    ])
    
    chain = final_report_prompt | llm | StrOutputParser()
    report = await chain.ainvoke({
        "plan": state["report_plan"],
        "learnings": learnings_str,
        "sources": sources_str,
        "images": images_str
    })
    
    return {"final_report": report}

def route_to_search(state: DeepResearchState):
    """
    Conditional Edge: Routes from generate_queries to perform_search (fan-out).
    """
    return [Send("perform_search", task) for task in state["serp_queries"]]

