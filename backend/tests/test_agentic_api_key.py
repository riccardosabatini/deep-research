import os
import pytest
from unittest.mock import MagicMock
from deepresearch.llm import get_llm
from langchain_core.runnables import RunnableConfig

def test_get_llm_fallback():
    config = MagicMock()
    # Configure mock to behave like a dict/config
    config.get.return_value = {}
    
    # Test fallback to AI_API_KEY
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ["AI_API_KEY"] = "test-ai-key"
    
    # Test with specific model name (agentic style)
    llm = get_llm(config, model_name="openai:gpt-4")
    assert llm.openai_api_key.get_secret_value() == "test-ai-key"
    
    # Test with model type (interactive style)
    # Set env vars for Config.from_env() to pick up
    os.environ["THINKING_MODEL"] = "openai:gpt-4-thinking"
    
    llm = get_llm(config, model_type="thinking")
    assert llm.model_name == "gpt-4-thinking"
    assert llm.openai_api_key.get_secret_value() == "test-ai-key"

def test_get_llm_base_url():
    config = MagicMock()
    config.get.return_value = {}
    os.environ["AI_BASE_URL"] = "https://test.url"
    
    llm = get_llm(config, model_name="openai:gpt-4")
    assert llm.openai_api_base == "https://test.url"
