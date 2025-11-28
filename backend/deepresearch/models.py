from typing import List, Optional, TypedDict, Annotated
import operator
from pydantic import BaseModel, Field

# --- Data Models ---

class SearchResultItem(BaseModel):
    url: str
    title: str
    content: str

class ImageSource(BaseModel):
    url: str
    description: Optional[str] = None

class DeepResearchSearchTask(BaseModel):
    query: str
    research_goal: str = Field(..., description="The specific goal of this search query")

class DeepResearchQueryList(BaseModel):
    queries: List[DeepResearchSearchTask]

class DeepResearchSearchResult(BaseModel):
    query: str
    research_goal: str
    learnings: List[str]
    sources: List[SearchResultItem] = []
    images: List[ImageSource] = []

class FinalReportResult(BaseModel):
    title: str
    final_report: str
    learnings: List[str]
    sources: List[SearchResultItem]
    images: List[ImageSource]

# --- Graph State ---

class DeepResearchState(TypedDict):
    query: str
    report_plan: str
    serp_queries: List[DeepResearchSearchTask]
    # Annotated with operator.add to allow appending results from parallel nodes
    search_results: Annotated[List[DeepResearchSearchResult], operator.add]
    user_feedback: Optional[str]
    final_report: str
