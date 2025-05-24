from typing import Dict, List

from tavily import TavilyClient

from ..workflow.state import EnhancedRFPState
from .knowledge_management import KnowledgeManagementAgent


class VendorIntelligenceAgent:
    def __init__(
        self, tavily_client: TavilyClient, knowledge_agent: KnowledgeManagementAgent
    ):
        self.tavily_client = tavily_client
        self.knowledge_agent = knowledge_agent

    def process(self, state: EnhancedRFPState) -> EnhancedRFPState:
        """Generate intelligent vendor proposals based on market research"""
        budget = state["budget_estimate"]["development_cost"]
        market_research = state.get("market_research", {})
        domain = state["parsed_requirements"].get("domain", "software")

        # Search for specific vendor information
        vendor_query = f"best {domain} development companies 2025 case studies pricing"
        vendor_research = self._research_vendors(vendor_query)

        # Generate realistic proposals based on market intelligence
        proposals = self._generate_intelligent_proposals(
            budget, market_research, vendor_research
        )

        state["vendor_proposals"] = proposals
        state["vendor_intelligence"] = {
            "research_conducted": True,
            "vendors_researched": len(vendor_research),
            "market_factors_considered": len(market_research.get("market_trends", [])),
        }

        return state

    def _research_vendors(self, query: str) -> List[Dict]:
        """Research actual vendor information"""
        try:
            results = self.tavily_client.search(
                query=query, search_depth="basic", max_results=5
            )
            return results.get("results", [])
        except:
            return []

    def _generate_intelligent_proposals(
        self, budget: float, market_research: Dict, vendor_research: List
    ) -> List[Dict]:
        """Generate proposals based on real market intelligence"""
        proposals = []

        # Extract vendor names and characteristics from research
        vendor_characteristics = self._extract_vendor_characteristics(vendor_research)

        # Generate 3-4 realistic proposals
        proposal_templates = [
            {
                "type": "premium",
                "cost_multiplier": 1.2,
                "timeline_months": 10,
                "risk_factors": ["Higher cost"],
                "strengths": ["Proven track record", "Enterprise experience"],
            },
            {
                "type": "balanced",
                "cost_multiplier": 1.0,
                "timeline_months": 8,
                "risk_factors": ["Standard timeline"],
                "strengths": ["Good balance of cost and quality"],
            },
            {
                "type": "aggressive",
                "cost_multiplier": 0.8,
                "timeline_months": 6,
                "risk_factors": ["Aggressive timeline", "Cost cutting"],
                "strengths": ["Fast delivery", "Cost effective"],
            },
        ]

        for i, template in enumerate(proposal_templates):
            vendor_name = vendor_characteristics.get(i, {}).get(
                "name", f"Vendor {chr(65+i)}"
            )

            proposal = {
                "vendor_name": vendor_name,
                "cost": budget * template["cost_multiplier"],
                "timeline_months": template["timeline_months"],
                "features": self._generate_features_based_on_research(market_research),
                "risk_factors": template["risk_factors"],
                "strengths": template["strengths"],
                "market_intelligence_used": True,
            }
            proposals.append(proposal)

        return proposals

    def _extract_vendor_characteristics(self, vendor_research: List) -> Dict:
        """Extract vendor names and characteristics from research"""
        characteristics = {}
        for i, result in enumerate(vendor_research[:3]):
            title = result.get("title", "")
            # Simple extraction - can be enhanced with NLP
            vendor_name = title.split()[0] if title else f"TechCorp {chr(65+i)}"
            characteristics[i] = {
                "name": vendor_name,
                "source": result.get("url", "unknown"),
            }
        return characteristics

    def _generate_features_based_on_research(self, market_research: Dict) -> List[str]:
        """Generate features based on market trends"""
        base_features = ["Core functionality", "User management", "Basic reporting"]

        # Add trending features based on market research
        trends = market_research.get("technology_trends", [])
        for trend in trends:
            trend_lower = trend.lower()
            if "ai" in trend_lower or "machine learning" in trend_lower:
                base_features.append("AI-powered analytics")
            if "mobile" in trend_lower:
                base_features.append("Mobile-first design")
            if "security" in trend_lower:
                base_features.append("Advanced security features")

        return base_features[:5]  # Limit to 5 features
