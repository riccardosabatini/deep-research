import pytest
from unittest.mock import AsyncMock, MagicMock
from src.nodes import plan_research, generate_queries, perform_search, write_report
from src.models import DeepResearchState, DeepResearchSearchTask, DeepResearchQueryList, DeepResearchSearchResult

@pytest.mark.asyncio
async def test_plan_research(mock_llm, mock_config, mocker):
    # Mock get_llm to return our mock_llm
    mocker.patch("src.nodes.get_llm", return_value=mock_llm)
    
    state = {"query": "Test Query"}
    result = await plan_research(state, mock_config)
    
    assert "report_plan" in result
    assert result["report_plan"] == "Mocked LLM Response"
    mock_llm.ainvoke.assert_called_once()

@pytest.mark.asyncio
async def test_generate_queries(mock_llm, mock_config, mocker):
    mocker.patch("src.nodes.get_llm", return_value=mock_llm)
    
    # Setup mock for structured output
    mock_queries = DeepResearchQueryList(queries=[
        DeepResearchSearchTask(query="q1", research_goal="g1"),
        DeepResearchSearchTask(query="q2", research_goal="g2")
    ])
    mock_llm.with_structured_output.return_value.ainvoke.return_value = mock_queries
    
    state = {"report_plan": "Test Plan"}
    result = await generate_queries(state, mock_config)
    
    assert "serp_queries" in result
    assert len(result["serp_queries"]) == 2
    assert result["serp_queries"][0]["query"] == "q1"

@pytest.mark.asyncio
async def test_perform_search(mock_llm, mock_search_tools, mock_config, mocker):
    mocker.patch("src.nodes.get_llm", return_value=mock_llm)
    mocker.patch("src.nodes.search_tools", mock_search_tools)
    
    # Mock DB functions
    mocker.patch("src.nodes.get_search_result", return_value=None) # Cache miss
    mocker.patch("src.nodes.save_search_result", new_callable=AsyncMock)
    
    task = {"query": "Test Search", "research_goal": "Test Goal"}
    result = await perform_search(task, mock_config)
    
    assert "search_results" in result
    assert len(result["search_results"]) == 1
    search_res = result["search_results"][0]
    assert search_res["query"] == "Test Search"
    assert len(search_res["sources"]) == 2
    assert len(search_res["images"]) == 1
    
    mock_search_tools.perform_search.assert_called_once_with("Test Search")

@pytest.mark.asyncio
async def test_write_report(mock_llm, mock_config, mocker):
    mocker.patch("src.nodes.get_llm", return_value=mock_llm)
    
    state = {
        "report_plan": "Test Plan",
        "search_results": [
            {
                "query": "q1", 
                "research_goal": "g1", 
                "learnings": ["l1"], 
                "sources": [], 
                "images": []
            }
        ]
    }
    result = await write_report(state, mock_config)
    
    assert "final_report" in result
    assert result["final_report"] == "Mocked LLM Response"
