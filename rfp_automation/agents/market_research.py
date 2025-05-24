import time
from datetime import datetime
from typing import Dict, List

from langchain_community.retrievers import TavilySearchAPIRetriever
from tavily import TavilyClient

from ..workflow.state import EnhancedRFPState
from ..config.settings import get_settings
from ..utils.llm import get_llm

from pydantic import BaseModel, Field
from typing import List

# Settings and LLM setup
settings = get_settings()


class MarketInsightsModel(BaseModel):
    market_trends: List[str] = Field(..., description="Key market trends")
    vendor_landscape: List[str] = Field(..., description="Top vendor-related insights")
    technology_trends: List[str] = Field(
        ..., description="Emerging technology practices"
    )


class MarketResearchAgent:
    def __init__(self, tavily_api_key: str):
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.retriever = TavilySearchAPIRetriever(
            api_key=tavily_api_key, k=5, include_raw_content=True
        )
        self.llm_generate = get_llm(
            temperature=0.7,
            max_tokens=1024,
            response_model=MarketInsightsModel,
        )

    def process(self, state: EnhancedRFPState) -> EnhancedRFPState:
        parsed = state["parsed_requirements"]
        domain = parsed.get("domain", "software")
        scale = parsed.get("scale", "1000")
        platform = parsed.get("platform", "cloud")
        features = parsed.get("features", "core functionality")

        try:
            # Build queries
            market_query = f"{domain} {platform} development cost for {scale} users with {features} in 2025"
            vendor_query = f"top {domain} {platform} solution providers with {features} support 2025"
            tech_query = f"technology trends 2025 for {domain} {platform} platforms using {features}"

            # Tavily search
            market_results = self._search_with_retry(market_query)
            vendor_results = self._search_with_retry(vendor_query)
            tech_results = self._search_with_retry(tech_query)

            # Compose input for LLM
            prompt = f"""
You are a market research analyst.

The user is looking for market research in the {domain} domain, focusing on the {platform} platform, for about {scale} users, and is especially interested in {features}.

You are given the following search result excerpts:

--- MARKET ---
{self._combine_results(market_results)}

--- VENDORS ---
{self._combine_results(vendor_results)}

--- TECHNOLOGY ---
{self._combine_results(tech_results)}

Please return structured insights as:
- market_trends: list of trends in pricing, growth, or demand
- vendor_landscape: list of top vendors and their strengths or weaknesses
- technology_trends: trends in tooling, architecture, best practices
"""

            insights = self.llm_generate(prompt)

            market_research = {
                "market_trends": insights.market_trends,
                "vendor_landscape": insights.vendor_landscape,
                "technology_trends": insights.technology_trends,
                "search_timestamp": datetime.now().isoformat(),
                "queries_performed": [market_query, vendor_query, tech_query],
            }

            state["market_research"] = market_research
            state["search_metadata"] = {
                "total_searches": 3,
                "search_duration": "real-time",
                "sources_found": len(market_results)
                + len(vendor_results)
                + len(tech_results),
            }

        except Exception as e:
            state["market_research"] = {
                "error": str(e),
                "fallback_used": True,
                "market_trends": [
                    "Cloud-native architecture trending",
                    "Microservices adoption increasing",
                ],
                "vendor_landscape": [
                    "Established players dominate",
                    "Emerging startups offer innovation",
                ],
                "technology_trends": [
                    "AI integration standard",
                    "Security-first development",
                ],
            }

        return state

    def _search_with_retry(self, query: str, max_retries: int = 3) -> List[Dict]:
        for attempt in range(max_retries):
            try:
                results = self.tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=5,
                    include_domains=[
                        "techcrunch.com",
                        "stackoverflow.com",
                        "github.com",
                    ],
                    exclude_domains=["spam-site.com"],
                )
                return results.get("results", [])
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2**attempt)
        return []

    def _combine_results(self, results: List[Dict]) -> str:
        return "\n\n".join(
            f"Title: {r.get('title', '')}\nContent: {r.get('content', '')[:500]}"
            for r in results[:3]
        )
