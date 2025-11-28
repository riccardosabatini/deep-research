import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

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
    log_level: str = "INFO"
    db_provider: str = "sqlite" # sqlite or postgres
    db_uri: str = "sqlite:///search_results.db" # filename for sqlite, connection string for postgres
    max_search_results: int = 5
    
    @classmethod
    def from_env(cls):
        return cls(
            thinking_model=os.getenv("THINKING_MODEL", "gpt-4o"),
            task_model=os.getenv("TASK_MODEL", "gpt-4o"),
            provider=os.getenv("AI_PROVIDER", "openai"),
            base_url=os.getenv("AI_BASE_URL", ""),
            api_key=os.getenv("AI_API_KEY", ""),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            db_provider=os.getenv("DB_PROVIDER", "sqlite").lower(),
            db_uri=os.getenv("DB_URI", "checkpoints.db")
        )
