from typing import Dict, List

from ..workflow.state import EnhancedRFPState


class RecommendationEngine:
    def process(self, state: EnhancedRFPState) -> EnhancedRFPState:
        """Rank proposals and generate recommendations"""
        proposals = state["proposal_scores"]

        # Calculate weighted scores
        for proposal in proposals:
            score = self._calculate_weighted_score(proposal)
            proposal["final_score"] = score

        # Sort by score (higher is better)
        ranked_proposals = sorted(
            proposals, key=lambda x: x["final_score"], reverse=True
        )

        recommendations = {
            "top_choice": ranked_proposals[0],
            "ranked_proposals": ranked_proposals,
            "summary": self._generate_summary(ranked_proposals),
        }

        state["recommendations"] = recommendations
        return state

    def _calculate_weighted_score(self, proposal: Dict) -> float:
        """Calculate weighted score (0-100)"""
        # Cost score (inverse - lower cost is better)
        cost_score = max(0, 100 - (proposal["cost"] / 1000))

        # Risk score (inverse - lower risk is better)
        risk_score = (1 - proposal["risk_score"]) * 100

        # Feature score
        feature_score = len(proposal["features"]) * 20

        # Timeline score (reasonable timeline gets higher score)
        timeline_score = 100 if 8 <= proposal["timeline_months"] <= 12 else 50

        # Weighted average
        weighted_score = (
            cost_score * 0.3
            + risk_score * 0.3
            + feature_score * 0.2
            + timeline_score * 0.2
        )

        return weighted_score

    def _generate_summary(self, proposals: List[Dict]) -> str:
        top = proposals[0]
        return f"Recommended vendor: {top['vendor_name']} with score {top['final_score']:.1f}/100"
