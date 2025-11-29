import os
from typing import List, Dict, Any
from tavily import TavilyClient
from .configuration import Config
from .models import SearchResultItem, ImageSource

import json
from redis import Redis
from exa_py import Exa

class SearchTools:
    def __init__(self):
        self.config = Config.from_env()
        
        # Initialize Tavily
        if self.config.search_provider == "tavily":
            if not self.config.search_api_key:
                raise ValueError("SEARCH_API_KEY environment variable is not set")
            self.client = TavilyClient(api_key=self.config.search_api_key)
            
        # Initialize Exa
        elif self.config.search_provider == "exa":
            if not self.config.search_api_key:
                raise ValueError("SEARCH_API_KEY environment variable is not set")
            self.exa_client = Exa(api_key=self.config.search_api_key)
            
        self.redis_client = None
        if self.config.redis_enabled:
            self.redis_client = Redis.from_url(self.config.redis_url)

    def _cache_key(self, query: str) -> str:
        return f"{self.config.search_provider}:{query}"
    
    async def perform_search(self, query: str) -> Dict[str, List[Any]]:
        """
        Executes a search using the configured provider and returns structured results.
        """
        # Check Redis Cache
        if self.redis_client:
            cache_key = self._cache_key(query)
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                print(f"[DEBUG] Cache hit for query: {query}")
                data = json.loads(cached_result)
                # Reconstruct objects
                sources = [SearchResultItem(**s) for s in data["sources"]]
                images = [ImageSource(**i) for i in data["images"]]
                return {"sources": sources, "images": images}

        try:
            sources = []
            images = []
            
            if self.config.search_provider == "tavily":
                response = self.client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=self.config.max_search_results,
                    include_images=True,
                    include_raw_content=True
                )
                
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

            elif self.config.search_provider == "exa":
                response = self.exa_client.search_and_contents(
                    query,
                    category=self.config.search_category,
                    context=True,
                    num_results=self.config.max_search_results,
                    text=True,
                    type="deep"
                )
                
                # Process Exa results
                # Exa results structure: response.results is a list of objects
                if hasattr(response, "results"):
                    for result in response.results:
                        sources.append(SearchResultItem(
                            title=result.title or "",
                            url=result.url or "",
                            content=result.text or ""
                        ))
                
                # Exa doesn't provide images in the same way, leaving empty for now
                
            result_data = {
                "sources": sources,
                "images": images
            }
            
            # Save to Redis Cache
            if self.redis_client:
                cache_key = self._cache_key(query)
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
