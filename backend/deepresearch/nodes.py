import asyncio
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
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
    review_prompt,
    auto_feedback_prompt
)
from .tools import SearchTools
from .configuration import Config
from .results_db import get_search_result, save_search_result, save_report

# Initialize
search_tools = SearchTools()
console = Console()

def get_llm(config: RunnableConfig, model_type: str = "task"):
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
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model_name=model_name,
            temperature=0,
            api_key=api_key if api_key else None,
            base_url=base_url if base_url else None
        )
    elif provider == "google" or provider == "google_genai":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            google_api_key=api_key if api_key else None
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
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
        api_key=api_key if api_key else None,
        max_retries=10 # Increase default retries for rate limits
    )

async def plan_research(state: DeepResearchState, config: RunnableConfig):
    """
    Node: Generates the research plan based on the user query.
    Uses: Thinking Model
    """
    console.print(Panel(f"Planning Research for: {state['query']}", title="[bold blue]Plan Research[/bold blue]"))
    llm = get_llm(config, "thinking")
    chain = (report_plan_prompt | llm | StrOutputParser()).with_retry(stop_after_attempt=5)
    plan = await chain.ainvoke({"query": state["query"]})
    
    console.print(Panel(f"Research Plan Generated: {plan}", title="[bold blue]Research Plan[/bold blue]"))

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
    chain = (serp_queries_prompt | structured_llm).with_retry(stop_after_attempt=5)
    
    result = await chain.ainvoke({"plan": state["report_plan"]})
    return {"serp_queries": [q.model_dump() for q in result.queries]}

async def perform_search(task: dict, config: RunnableConfig):
    """
    Node: Executes a single search task and processes the results.
    Uses: Task Model (for processing results)
    """
    # Get thread_id for scoping results
    configurable = config.get("configurable", {})
    thread_id = configurable.get("thread_id", "default")
    
    # Convert dict to model
    task_model = DeepResearchSearchTask(**task)
    
    # Check cache first
    cached = await get_search_result(thread_id, task_model.query)

    if cached:
        console.print(f"[dim]Found cached result for: {task_model.query}[/dim]")
        sources = [SearchResultItem(**s) for s in cached["raw_result"]["sources"]]
        images = [ImageSource(**i) for i in cached["raw_result"]["images"]]
        learnings = cached["learnings"]
        
        result = DeepResearchSearchResult(
            query=task_model.query,
            research_goal=task_model.research_goal,
            learnings=[learnings],
            sources=sources,
            images=images
        )
        return {"search_results": [result.model_dump()]}

    console.print(f"[green]Executing Search:[/green] {task_model.query}")
    llm = get_llm(config, "task")
    
    # 1. Execute Search
    search_data = await search_tools.perform_search(task_model.query)
    sources: List[SearchResultItem] = search_data["sources"]
    images: List[ImageSource] = search_data["images"]
    
    # 2. Process Results (Synthesize Learnings)
    # Format context for the prompt
    context_str = "\n\n".join([
        f"<content id=\"{s.id}\" url=\"{s.url}\">\n{s.content}\n</content>"
        for s in sources
    ])
    
    # Add retry logic with exponential backoff for rate limits
    chain = (process_search_result_prompt | llm | StrOutputParser()).with_retry(
        stop_after_attempt=10,
        wait_exponential_jitter=True
    )
    
    learnings = await chain.ainvoke({
        "query": task_model.query,
        "researchGoal": task_model.research_goal,
        "context": context_str
    })
    
    # Save to DB
    await save_search_result(
        research_id=thread_id,
        query=task_model.query,
        raw_result={"sources": [s.model_dump() for s in sources], "images": [i.model_dump() for i in images]},
        learnings=learnings
    )
    
    # 3. Return Result
    result = DeepResearchSearchResult(
        query=task_model.query,
        research_goal=task_model.research_goal,
        learnings=[learnings],
        sources=sources,
        images=images
    )
    
    return {"search_results": [result.model_dump()]}

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
        # result is dict
        all_learnings.extend(result["learnings"])
    learnings_str = "\n".join(all_learnings)
    
    structured_llm = llm.with_structured_output(DeepResearchQueryList)
    chain = review_prompt | structured_llm
    
    result = await chain.ainvoke({
        "plan": state["report_plan"],
        "learnings": learnings_str,
        "suggestion": state["user_feedback"]
    })
    
    return {
        "serp_queries": [q.model_dump() for q in result.queries], 
        "user_feedback": None 
    }

async def analyze_research_gaps(state: DeepResearchState, config: RunnableConfig):
    """
    Node: Analyzes research gaps and generates feedback (Auto-Mode).
    Uses: Thinking Model
    """
    console.print(Panel("Analyzing Research Gaps...", title="[bold blue]Auto Feedback[/bold blue]"))
    llm = get_llm(config, "thinking")
    
    # Aggregate learnings
    all_learnings = []
    for result in state["search_results"]:
        all_learnings.extend(result["learnings"])
    learnings_str = "\n".join(all_learnings)
    
    chain = auto_feedback_prompt | llm | StrOutputParser()
    
    feedback = await chain.ainvoke({
        "plan": state["report_plan"],
        "learnings": learnings_str
    })
    
    if "SATISFIED" in feedback:
        console.print("[bold green]Auto-Feedback: Research Satisfied[/bold green]")
        return {"user_feedback": None, "feedback_loop_count": state.get("feedback_loop_count", 0) + 1}
    else:
        console.print(f"[bold yellow]Auto-Feedback Generated:[/bold yellow] {feedback}")
        return {"user_feedback": feedback, "feedback_loop_count": state.get("feedback_loop_count", 0) + 1}

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
        # result is dict
        all_learnings.extend(result["learnings"])
        all_sources.extend([SearchResultItem(**s) for s in result["sources"]])
        all_images.extend([ImageSource(**i) for i in result["images"]])
        
    # Format for prompt
    learnings_str = "\n".join(all_learnings)
    
    # sources_str = "\n".join([
    #     f"<source id=\"{s.id}\" url=\"{s.url}\">\n{s.title}\n</source>"
    #     for s in all_sources
    # ])
    
    # images_str = "\n".join([
    #     f"{i+1}. ![{img.description}]({img.url})"
    #     for i, img in enumerate(all_images)
    # ])
    
    chain = final_report_prompt | llm | StrOutputParser()
    report_pages = state.get("report_pages", Config.from_env().report_pages)
    report = await chain.ainvoke({
        "plan": state["report_plan"],
        "learnings": learnings_str,
        "report_pages": report_pages
    })
    
    import re
    import uuid

    source_dict = {s.id: s for s in all_sources}
    
    # Extract all the [uuid4] present in the report, in the order of appearance making sure only uuid4 are picked up
    raw_uuids = re.findall(r'\[(.*?)\]', report)
    uuids = []
    for uid_str in raw_uuids:
        try:
            # Validate if it's a valid UUID4
            if uuid.UUID(uid_str, version=4):
                uuids.append(uid_str)
        except ValueError:
            # Not a valid UUID, skip
            pass
    
    # Replace all the [uuid4] with the location of the uuid in the uuids list
    for i, uuid in enumerate(uuids):
        report = report.replace(f'[{uuid}]', f'[{i+1}]')
    
    # Append Sources Section
    sources_section = "\n\n## Sources\n\n" + "\n".join([
        f"- [{i+1}] [{source_dict[uuid].title}]({source_dict[uuid].url})"
        for i, uuid in enumerate(uuids)
    ])
    report += sources_section
    
    # Save to DB
    configurable = config.get("configurable", {})
    thread_id = configurable.get("thread_id", "default")
    await save_report(thread_id, report)
    
    return {"final_report": report}

def route_to_search(state: DeepResearchState):
    """
    Conditional Edge: Routes from generate_queries to perform_search (fan-out).
    """
    return [Send("perform_search", task) for task in state["serp_queries"]]
