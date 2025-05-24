import time
from datetime import datetime
from typing import Dict, List

from langchain_community.retrievers import TavilySearchAPIRetriever
from tavily import TavilyClient

from ..workflow.state import EnhancedRFPState


class MarketResearchAgent:
    def __init__(self, tavily_api_key: str):

        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.retriever = TavilySearchAPIRetriever(
            api_key=tavily_api_key, k=5, include_raw_content=True
        )

    def process(self, state: EnhancedRFPState) -> EnhancedRFPState:
        """Conduct market research using Tavily Search"""
        parsed = state["parsed_requirements"]
        domain = parsed.get("domain", "software")
        scale = parsed.get("scale", 1000)

        try:
            # Search for market trends and pricing
            market_query = (
                f"{domain} software development costs 2025 {scale} users pricing"
            )
            market_results = self._search_with_retry(market_query)

            # Search for vendor landscape
            vendor_query = f"top {domain} software development companies 2025 reviews"
            vendor_results = self._search_with_retry(vendor_query)

            # Search for technology trends
            tech_query = f"{domain} technology stack trends 2025 best practices"
            tech_results = self._search_with_retry(tech_query)

            market_research = {
                "market_trends": self._extract_insights(market_results),
                "vendor_landscape": self._extract_insights(vendor_results),
                "technology_trends": self._extract_insights(tech_results),
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
            # Fallback to cached data or default values
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
        """Search with retry logic and rate limiting"""
        for attempt in range(max_retries):
            try:
                results = self.tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=3,
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
                time.sleep(2**attempt)  # Exponential backoff
        return []

    def _extract_insights(self, search_results: List[Dict]) -> List[str]:
        """Extract key insights from search results"""
        insights = []
        for result in search_results[:3]:  # Top 3 results
            content = result.get("content", "")
            title = result.get("title", "")

            # Simple insight extraction (can be enhanced with NLP)
            if any(
                keyword in content.lower() for keyword in ["cost", "price", "budget"]
            ):
                insights.append(f"Market insight from {title}: {content[:200]}...")

        return insights or ["No specific market insights found"]
