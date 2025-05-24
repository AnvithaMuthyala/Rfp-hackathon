from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from ..utils.llm import get_llm
from ..workflow.state import EnhancedRFPState


class ContactInfo(BaseModel):
    name: str
    designation: str
    department: str
    email: str
    phone: str


class EligibilityCriterion(BaseModel):
    description: str
    documents_required: List[str]


class BidSubmission(BaseModel):
    submission_method: str
    submission_deadline: str
    bid_opening_date: str
    bid_validity_period: int  # in days


class EvaluationCriteria(BaseModel):
    technical_weight: int
    financial_weight: int
    evaluation_method: str  # e.g., QCBS


class SecurityRequirements(BaseModel):
    data_encryption: bool
    compliance_standards: List[str]
    audit_trail: bool


class BudgetEstimate(BaseModel):
    estimated_cost: float
    currency: str
    funding_source: Optional[str] = None


class RFP(BaseModel):
    title: str
    project_name: str
    issuing_authority: str
    issuing_department: str
    contact_info: ContactInfo
    eligibility_criteria: List[EligibilityCriterion]
    scope_of_work: str
    deliverables: List[str]
    timeline: str
    budget_estimate: BudgetEstimate
    bid_submission: BidSubmission
    evaluation_criteria: EvaluationCriteria
    security_requirements: SecurityRequirements
    annexures: Optional[List[str]]
    publication_date: str
    last_date_for_submission: str
    pre_bid_meeting_date: Optional[str]
    corrigendum_details: Optional[str]


class RFPGeneratorAgent:

    def __init__(self):
        self.llm = get_llm(response_model=RFP)

    def process(self, state: EnhancedRFPState) -> EnhancedRFPState:
        """Generate structured RFP document"""
        reqs = state["aggregated_requirements"]

        rfp = self.llm(f"Generate RFP document with these inputs: {str(state)}")

        llm = get_llm()
        markdown_document = llm(
            f"Generate a detailed markdown RFP document using these: {rfp.model_dump()}"
        )

        state["rfp_document"] = markdown_document.content
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
