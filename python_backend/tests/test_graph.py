import pytest
from unittest.mock import AsyncMock, MagicMock
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage
from src.graph import workflow
from src.models import DeepResearchState, DeepResearchSearchTask, DeepResearchQueryList, DeepResearchSearchResult

@pytest.mark.asyncio
async def test_graph_execution(mock_llm, mock_search_tools, mock_config, mocker):
    # ... (keep mocks) ...
    mocker.patch("src.nodes.get_llm", return_value=mock_llm)
    mocker.patch("src.nodes.search_tools", mock_search_tools)
    mocker.patch("src.nodes.get_search_result", return_value=None)
    mocker.patch("src.nodes.save_search_result", new_callable=AsyncMock)
    
    # Setup mock responses
    mock_plan = AIMessage(content="Mocked Plan")
    mock_learnings = AIMessage(content="Mocked Learnings")
    mock_report = AIMessage(content="Mocked Report")
    
    # side_effect for multiple calls: plan_research, perform_search, write_report
    # We have 1 plan_research call.
    # We have 1 perform_search call (for 1 query).
    # We have 1 write_report call.
    # Total 3 calls to ainvoke (plus structured output calls which are separate).
    mock_llm.ainvoke.side_effect = [mock_plan, mock_learnings, mock_report]
    mock_llm.invoke.side_effect = [mock_plan, mock_learnings, mock_report]
    mock_llm.return_value = mock_plan # Default fallback
    
    mock_queries = DeepResearchQueryList(queries=[
        DeepResearchSearchTask(query="q1", research_goal="g1")
    ])
    mock_llm.with_structured_output.return_value.ainvoke.return_value = mock_queries
    
    # Use MemorySaver
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer, interrupt_before=["review_step"])
    
    initial_state = {
        "query": "Test Query",
        "report_plan": "",
        "serp_queries": [],
        "search_results": [],
        "user_feedback": None,
        "final_report": ""
    }
    
    thread_id = "test_thread"
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run until interrupt
    async for event in app.astream(initial_state, config=config):
        pass
        
    # Check state at interrupt
    state = await app.aget_state(config)
    assert state.values["report_plan"] == "Mocked Plan"
    assert len(state.values["serp_queries"]) == 1
    assert len(state.values["search_results"]) == 1
    assert state.next == ("review_step",)
    
    # Resume with no feedback (user_feedback defaults to None)
    async for event in app.astream(None, config=config):
        pass
        
    # Check final state
    state = await app.aget_state(config)
    assert "final_report" in state.values
    assert state.values["final_report"] == "Mocked Report"
