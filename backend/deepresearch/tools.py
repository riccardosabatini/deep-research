import os
from typing import List, Dict, Any
from tavily import TavilyClient
from .configuration import Config
from .models import SearchResultItem, ImageSource

import json
from redis import Redis

class SearchTools:
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable is not set")
        self.client = TavilyClient(api_key=api_key)
        self.config = Config.from_env()
        self.redis_client = None
        if self.config.redis_enabled:
            self.redis_client = Redis.from_url(self.config.redis_url)

    async def perform_search(self, query: str) -> Dict[str, List[Any]]:
        """
        Executes a search using Tavily and returns structured results.
        """
        # Check Redis Cache
        if self.redis_client:
            cache_key = f"tavily:{query}"
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                print(f"[DEBUG] Cache hit for query: {query}")
                data = json.loads(cached_result)
                # Reconstruct objects
                sources = [SearchResultItem(**s) for s in data["sources"]]
                images = [ImageSource(**i) for i in data["images"]]
                return {"sources": sources, "images": images}

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
                if isinstance(img, str):
                    images.append(ImageSource(url=img, description=None))
                elif isinstance(img, dict):
                    images.append(ImageSource(url=img.get("url"), description=img.get("description")))

            result_data = {
                "sources": sources,
                "images": images
            }
            
            # Save to Redis Cache
            if self.redis_client:
                cache_key = f"tavily:{query}"
                # Serialize objects to dicts for JSON
                cache_data = {
                    "sources": [s.model_dump() for s in sources],
                    "images": [i.model_dump() for i in images]
                }
                self.redis_client.set(cache_key, json.dumps(cache_data), ex=86400) # Cache for 24h

            return result_data
            
        except Exception as e:
            print(f"Error performing search for '{query}': {e}")
            return {"sources": [], "images": []}
