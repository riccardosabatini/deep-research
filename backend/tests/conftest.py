import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage
from deepresearch.interactive.models import SearchResultItem, ImageSource

from langchain_openai import ChatOpenAI

@pytest.fixture
def mock_llm():
    """
    Mocks the ChatOpenAI client.
    """
    mock = MagicMock(spec=ChatOpenAI)
    # Mock ainvoke to return an AIMessage
    mock.ainvoke = AsyncMock(return_value=AIMessage(content="Mocked LLM Response"))
    # Mock invoke (sync) just in case
    mock.invoke = MagicMock(return_value=AIMessage(content="Mocked LLM Response"))
    
    # Mock with_structured_output to return a mock that returns a Pydantic model
    structured_llm = MagicMock(spec=ChatOpenAI)
    structured_llm.ainvoke = AsyncMock()
    mock.with_structured_output.return_value = structured_llm
    
    return mock

@pytest.fixture
def mock_search_tools():
    """
    Mocks the SearchTools class.
    """
    mock = MagicMock()
    mock.perform_search = AsyncMock(return_value={
        "sources": [
            SearchResultItem(url="http://example.com/1", title="Example 1", content="Content 1"),
            SearchResultItem(url="http://example.com/2", title="Example 2", content="Content 2")
        ],
        "images": [
            ImageSource(url="http://example.com/image1.jpg", description="Image 1")
        ]
    })
    return mock

@pytest.fixture
def mock_config():
    """
    Mocks the RunnableConfig.
    """
    return {"configurable": {"thread_id": "test_thread"}}
