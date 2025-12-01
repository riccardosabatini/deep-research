from typing import Optional
from pydantic import BaseModel, Field
import shortuuid

# --- Data Models ---

class SearchResultItem(BaseModel):
    id: str = Field(default_factory=lambda: shortuuid.ShortUUID().random(length=8))
    url: str
    title: str
    content: str

class ImageSource(BaseModel):
    url: str
    description: Optional[str] = None
