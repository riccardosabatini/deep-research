import importlib
from typing import Any

def get_prompts(prompt_set: str = "default") -> Any:
    """
    Dynamically imports and returns the specified prompt set module.
    """
    try:
        module = importlib.import_module(f".{prompt_set}", package=__name__)
        return module
    except ImportError:
        raise ValueError(f"Prompt set '{prompt_set}' not found.")
