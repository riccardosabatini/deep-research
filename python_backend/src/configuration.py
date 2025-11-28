import os
from dataclasses import dataclass

@dataclass
class Config:
    """
    Configuration for the Deep Research process.
    """
    thinking_model: str = "gpt-4o"
    task_model: str = "gpt-4o"
    provider: str = "openai"
    base_url: str = ""
    api_key: str = ""
    
    @classmethod
    def from_env(cls):
        return cls(
            thinking_model=os.getenv("THINKING_MODEL", "gpt-4o"),
            task_model=os.getenv("TASK_MODEL", "gpt-4o"),
            provider=os.getenv("AI_PROVIDER", "openai"),
            base_url=os.getenv("AI_BASE_URL", ""),
            api_key=os.getenv("AI_API_KEY", "")
        )
