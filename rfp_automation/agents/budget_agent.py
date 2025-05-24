from typing import Dict, List

from ..workflow.state import EnhancedRFPState


class EnhancedBudgetAgent:
    def process(self, state: EnhancedRFPState) -> EnhancedRFPState:
        """Enhanced budget estimation using market research"""
        parsed = state["parsed_requirements"]
        market_research = state.get("market_research", {})
        cached_knowledge = state.get("cached_knowledge", [])

        # Base calculation
        base_cost = 50000
        scale_multiplier = parsed.get("scale", 1000) / 1000
        platform_multiplier = len(parsed.get("platform", ["web"])) * 0.5 + 0.5

        # Market adjustment factor from research
        market_adjustment = self._calculate_market_adjustment(
            market_research, cached_knowledge
        )

        estimated_cost = (
            base_cost * scale_multiplier * platform_multiplier * market_adjustment
        )

        budget_estimate = {
            "development_cost": estimated_cost,
            "monthly_hosting": estimated_cost * 0.02,
            "maintenance_yearly": estimated_cost * 0.2,
            "total_first_year": estimated_cost * 1.44,
            "market_adjustment_factor": market_adjustment,
            "market_insights_used": len(market_research.get("market_trends", [])),
            "cached_insights_used": len(cached_knowledge),
        }

        state["budget_estimate"] = budget_estimate
        return state

    def _calculate_market_adjustment(
        self, market_research: Dict, cached_knowledge: List
    ) -> float:
        """Calculate market-based adjustment factor"""
        adjustment = 1.0

        # Analyze market trends for cost indicators
        trends = market_research.get("market_trends", [])
        for trend in trends:
            trend_lower = trend.lower()
            if any(
                keyword in trend_lower for keyword in ["expensive", "costly", "premium"]
            ):
                adjustment += 0.1
            elif any(
                keyword in trend_lower
                for keyword in ["affordable", "cost-effective", "budget"]
            ):
                adjustment -= 0.05

        # Use cached knowledge for historical context
        for knowledge in cached_knowledge:
            if knowledge.get("relevance_score", 0) > 0.8:
                content = knowledge.get("content", "").lower()
                if "cost increase" in content or "price rise" in content:
                    adjustment += 0.05

        return max(0.7, min(1.5, adjustment))  # Cap between 70% and 150%
