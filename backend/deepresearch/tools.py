import os
from typing import List, Dict, Any
from tavily import TavilyClient
from .configuration import Config
from .models import SearchResultItem, ImageSource

class SearchTools:
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable is not set")
        self.client = TavilyClient(api_key=api_key)
        self.config = Config.from_env()

    async def perform_search(self, query: str) -> Dict[str, List[Any]]:
        """
        Executes a search using Tavily and returns structured results.
        """
        # Tavily's python client is synchronous by default, but we can wrap it or use their async client if available.
        # For now, we'll run it synchronously as the graph node will be async. 
        # Ideally, we'd use run_in_executor or an async client.
        
        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=self.config.max_search_results,
                include_images=True,
                include_raw_content=True
            )
            
            sources = []
            images = []
            
            # Process text results
            for result in response.get("results", []):
                sources.append(SearchResultItem(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    content=result.get("raw_content") or result.get("content", "")
                ))
                
            # Process images
            for img in response.get("images", []):
                # Tavily returns strings (urls) or objects depending on config.
                # Assuming simple list of strings based on typical usage, 
                # but let's handle if it's a dict just in case or if the API changed.
                if isinstance(img, str):
                    images.append(ImageSource(url=img, description=None))
                elif isinstance(img, dict):
                    images.append(ImageSource(url=img.get("url"), description=img.get("description")))

            return {
                "sources": sources,
                "images": images
            }
            
        except Exception as e:
            print(f"Error performing search for '{query}': {e}")
            return {"sources": [], "images": []}
