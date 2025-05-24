from typing import Any, Dict, List, TypedDict


class EnhancedRFPState(TypedDict):
    """Enhanced state with search and vector storage capabilities"""

    user_input: str
    parsed_requirements: Dict[str, Any]
    market_research: Dict[str, Any]  # New: Tavily search results
    vendor_intelligence: Dict[str, Any]  # New: Vendor research
    cached_knowledge: List[Dict[str, Any]]  # New: ChromaDB results
    suggestions: List[str]
    security_requirements: Dict[str, Any]
    budget_estimate: Dict[str, Any]
    tech_recommendations: Dict[str, Any]
    aggregated_requirements: Dict[str, Any]
    rfp_document: str
    user_approved: bool
    vendor_proposals: List[Dict[str, Any]]
    proposal_scores: List[Dict[str, Any]]
    recommendations: Dict[str, Any]
    audit_log: List[Dict[str, Any]]
    visual_summary: Dict[str, Any]
    reviewer_comments: List[str]
    analytics: Dict[str, Any]
    search_metadata: Dict[str, Any]  # New: Search performance metrics
