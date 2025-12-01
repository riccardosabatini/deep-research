import os
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from .configuration import Config

def get_llm(config: RunnableConfig, model_name: str = None, model_type: str = "task"):
    """
    Helper to get the LLM instance based on configuration.
    Supports both 'interactive' (via model_type) and 'agentic' (via specific model_name) modes.
    """
    # Load config from configurable or env
    configurable = config.get("configurable", {}) if config else {}
    env_config = Config.from_env()
    
    # Resolve model name
    if model_name:
        # If specific model name is provided (agentic mode usually passes this)
        pass
    elif model_type == "thinking":
        model_name = configurable.get("thinking_model", env_config.thinking_model)
    else:
        model_name = configurable.get("task_model", env_config.task_model)
        
    # Handle "provider:model" syntax (common in agentic config)
    provider = None
    if ":" in model_name:
        provider, model_name = model_name.split(":", 1)
    
    # If provider not parsed from name, get from config/env
    if not provider:
        provider = configurable.get("provider", env_config.provider).lower()
    
    # Get API key and Base URL
    # Priority: Configurable -> Env (specific) -> Env (generic AI_*)
    
    # We can use a helper to get keys based on provider
    api_key = None
    base_url = None
    
    # Check for specific keys in configurable
    if provider == "openai":
        api_key = configurable.get("openai_api_key")
    elif provider == "anthropic":
        api_key = configurable.get("anthropic_api_key")
    elif provider == "google":
        api_key = configurable.get("google_api_key")
        
    # Fallback to generic api_key in configurable
    if not api_key:
        api_key = configurable.get("api_key")
        
    # Fallback to Env
    if not api_key:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        elif provider == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
            
    # Fallback to generic AI_API_KEY
    if not api_key:
        api_key = os.getenv("AI_API_KEY") # or env_config.api_key
        
    # Base URL
    base_url = configurable.get("base_url", env_config.base_url)
    if not base_url:
        base_url = os.getenv("AI_BASE_URL")

    # Instantiate Model
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
    
    # Default to OpenAI
    return ChatOpenAI(
        model=model_name, 
        temperature=0,
        base_url=base_url if base_url else None,
        api_key=api_key if api_key else None,
        max_retries=10 
    )
