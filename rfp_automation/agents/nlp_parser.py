from datetime import datetime
import logging
from typing import Dict, List
from pydantic import BaseModel, Field

from ..workflow.state import EnhancedRFPState
from ..utils.llm import get_llm


class NLParserResponse(BaseModel):
    domain: str = Field(..., description="")
    scale: int = Field(..., description="")
    platform: list[str] = Field(..., description="")
    features: list[str] = Field(..., description="")
    urgency: str


class NLPParserAgent:
    def __init__(self):
        self.keywords = {
            "domain": ["logistics", "tracking", "mobile", "app", "web", "platform"],
            "scale": ["users", "scalable", "concurrent", "load"],
            "platform": ["mobile", "web", "desktop", "ios", "android"],
            "features": ["real-time", "analytics", "dashboard", "reporting"],
        }

        self.llm = get_llm(0, response_model=NLParserResponse)

    def process(self, state: EnhancedRFPState) -> EnhancedRFPState:
        """Extract structured requirements from user input"""
        user_input = state["user_input"].lower()

        prompt = f"""From the input, identify and extract the following:

- "domain": The industry or area the request pertains to (e.g., healthcare, e-commerce, education).
- "scale": The expected size or scope (e.g., startup-level, enterprise, global).
- "platform": The type of platform or environment (e.g., web, mobile, cloud).
- "features": A list of key features or functionalities mentioned or implied.

User input:
{user_input}"""

        response = self.llm(prompt)

        state["parsed_requirements"] = response.model_dump()  # type:ignore
        state["audit_log"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "agent": "NLPParser",
                "action": "parsed_requirements",
                "data": response.model_dump(),  # type:ignore
            }
        )

        return state


class SuggestionAgentResponse(BaseModel):
    suggestions: list[str]


class SuggestionAgent:

    def __init__(self):
        self.llm = get_llm(response_model=SuggestionAgentResponse)

    def process(self, state: EnhancedRFPState) -> EnhancedRFPState:
        """Generate smart suggestions for missing requirements"""
        knowledge = state["cached_knowledge"]
        parsed = state["parsed_requirements"]

        # Prepare a context summary from the knowledge base
        context = "\n".join([f"- {item['content']}" for item in knowledge])

        # Generate a prompt to the LLM
        prompt = (
            "You are a requirements analyst. Based on the following parsed requirements "
            "and contextual knowledge, suggest 3-5 questions or feature ideas the user "
            "might have missed.\n\n"
            f"Parsed Requirements:\n{parsed}\n\n"
            f"Contextual Knowledge:\n{context}\n\n"
            "Respond with a list of suggestions."
        )

        try:
            response = self.llm(prompt)
            state["suggestions"] = response.suggestions
        except Exception as e:
            # Fallback in case LLM fails
            state["suggestions"] = [
                "Could not generate suggestions due to an internal error."
            ]
            logging.exception("LLM failed during suggestion generation")

        return state


class SecurityAgent:
    def process(self, state: EnhancedRFPState) -> EnhancedRFPState:
        """Add security and compliance requirements"""
        parsed = state["parsed_requirements"]

        security_reqs = {
            "compliance": ["GDPR", "SOC2"],
            "authentication": "Multi-factor authentication required",
            "encryption": "End-to-end encryption for data in transit and at rest",
            "audit_logging": "Comprehensive audit logging required",
        }

        # Add specific requirements based on scale
        if parsed.get("scale", 0) > 10000:
            security_reqs["compliance"].append("ISO 27001")

        state["security_requirements"] = security_reqs
        return state


class BudgetAgent:
    def process(self, state: EnhancedRFPState) -> EnhancedRFPState:
        """Estimate budget using rule-based calculations"""
        parsed = state["parsed_requirements"]

        base_cost = 50000  # Base development cost
        scale_multiplier = parsed.get("scale", 1000) / 1000
        platform_multiplier = len(parsed.get("platform", ["web"])) * 0.5 + 0.5

        estimated_cost = base_cost * scale_multiplier * platform_multiplier

        budget_estimate = {
            "development_cost": estimated_cost,
            "monthly_hosting": estimated_cost * 0.02,
            "maintenance_yearly": estimated_cost * 0.2,
            "total_first_year": estimated_cost * 1.44,
        }

        state["budget_estimate"] = budget_estimate
        return state


class TechAgent:
    def process(self, state: EnhancedRFPState) -> EnhancedRFPState:
        """Suggest technology stack and architecture"""
        parsed = state["parsed_requirements"]

        tech_stack = {
            "backend": "Node.js/Python with microservices architecture",
            "database": "PostgreSQL with Redis for caching",
            "cloud": "AWS/Azure with auto-scaling groups",
            "monitoring": "CloudWatch/Application Insights",
        }

        if "mobile" in parsed.get("platform", []):
            tech_stack["mobile"] = "React Native or Flutter"

        if parsed.get("scale", 0) > 50000:
            tech_stack["load_balancer"] = "Application Load Balancer"
            tech_stack["cdn"] = "CloudFront/Azure CDN"

        state["tech_recommendations"] = tech_stack
        return state


class AggregatorAgent:
    def process(self, state: EnhancedRFPState) -> EnhancedRFPState:
        """Combine all agent outputs into unified requirements"""
        aggregated = {
            "parsed_requirements": state["parsed_requirements"],
            "security_requirements": state["security_requirements"],
            "budget_estimate": state["budget_estimate"],
            "tech_recommendations": state["tech_recommendations"],
            "suggestions_addressed": state["suggestions"],
        }

        state["aggregated_requirements"] = aggregated
        return state


class RFPGeneratorAgent:
    def process(self, state: EnhancedRFPState) -> EnhancedRFPState:
        """Generate structured RFP document"""
        reqs = state["aggregated_requirements"]

        rfp_template = f"""
# Request for Proposal (RFP)
## Project Overview
Domain: {reqs['parsed_requirements']['domain']}
Scale: {reqs['parsed_requirements']['scale']} users
Platform: {', '.join(reqs['parsed_requirements']['platform'])}

## Technical Requirements
{self._format_tech_requirements(reqs['tech_recommendations'])}

## Security & Compliance
{self._format_security_requirements(reqs['security_requirements'])}

## Budget Expectations
Development Cost: ${reqs['budget_estimate']['development_cost']:,.2f}
First Year Total: ${reqs['budget_estimate']['total_first_year']:,.2f}

## Evaluation Criteria
- Technical expertise (30%)
- Cost effectiveness (25%)
- Timeline feasibility (20%)
- Security compliance (15%)
- Past experience (10%)
        """

        state["rfp_document"] = rfp_template.strip()
        return state

    def _format_tech_requirements(self, tech_reqs: Dict) -> str:
        return "\n".join([f"- {k.title()}: {v}" for k, v in tech_reqs.items()])

    def _format_security_requirements(self, sec_reqs: Dict) -> str:
        formatted = []
        for k, v in sec_reqs.items():
            if isinstance(v, list):
                formatted.append(f"- {k.title()}: {', '.join(v)}")
            else:
                formatted.append(f"- {k.title()}: {v}")
        return "\n".join(formatted)


class VendorSimulatorAgent:
    def process(self, state: EnhancedRFPState) -> EnhancedRFPState:
        """Generate mock vendor proposals"""
        budget = state["budget_estimate"]["development_cost"]

        proposals = [
            {
                "vendor_name": "TechCorp Solutions",
                "cost": budget * 0.9,
                "timeline_months": 8,
                "features": ["Real-time tracking", "Mobile app", "Analytics dashboard"],
                "risk_factors": ["Aggressive timeline"],
            },
            {
                "vendor_name": "Reliable Systems Inc",
                "cost": budget * 1.1,
                "timeline_months": 12,
                "features": ["Real-time tracking", "Mobile app", "Advanced security"],
                "risk_factors": ["Higher cost"],
            },
            {
                "vendor_name": "StartupTech",
                "cost": budget * 0.7,
                "timeline_months": 6,
                "features": ["Basic tracking", "Web platform"],
                "risk_factors": ["Limited experience", "Unrealistic timeline"],
            },
        ]

        state["vendor_proposals"] = proposals
        return state


class RiskEvaluatorAgent:
    def process(self, state: EnhancedRFPState) -> EnhancedRFPState:
        """Evaluate and score proposals for risks"""
        proposals = state["vendor_proposals"]
        scored_proposals = []

        for proposal in proposals:
            risk_score = self._calculate_risk_score(proposal)
            scored_proposal = {**proposal, "risk_score": risk_score}
            scored_proposals.append(scored_proposal)

        state["proposal_scores"] = scored_proposals
        return state

    def _calculate_risk_score(self, proposal: Dict) -> float:
        """Calculate risk score (0-1, lower is better)"""
        risk_score = 0.0

        # Timeline risk
        if proposal["timeline_months"] < 8:
            risk_score += 0.3

        # Cost risk
        if proposal["cost"] < 50000:
            risk_score += 0.2

        # Feature completeness risk
        if len(proposal["features"]) < 3:
            risk_score += 0.2

        # Risk factors
        risk_score += len(proposal.get("risk_factors", [])) * 0.1

        return min(risk_score, 1.0)
