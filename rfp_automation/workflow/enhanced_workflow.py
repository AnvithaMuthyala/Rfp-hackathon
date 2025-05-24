import logging
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient

from ..agents import *
from .state import EnhancedRFPState

# Configure logging to file
log_file = "rfp_workflow.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),  # Optional: keep this to also see logs in the console
    ],
)
logger = logging.getLogger(__name__)


class EnhancedRFPAutomationWorkflow:
    def __init__(self, tavily_api_key: str, chroma_persist_dir: str = "./chroma_db"):
        # Initialize enhanced agents
        self.knowledge_agent = KnowledgeManagementAgent(chroma_persist_dir)
        self.market_research_agent = MarketResearchAgent(tavily_api_key)
        self.vendor_intelligence_agent = VendorIntelligenceAgent(
            TavilyClient(api_key=tavily_api_key), self.knowledge_agent
        )

        # Original agents (enhanced)
        self.agents = {
            "nlp_parser": NLPParserAgent(),
            "market_research": self.market_research_agent,
            "knowledge_management": self.knowledge_agent,
            "suggestion": SuggestionAgent(),
            "security": SecurityAgent(),
            "budget": EnhancedBudgetAgent(),
            "tech": TechAgent(),
            "aggregator": AggregatorAgent(),
            "rfp_generator": RFPGeneratorAgent(),
            "vendor_intelligence": self.vendor_intelligence_agent,
            "risk_evaluator": RiskEvaluatorAgent(),
            "recommendation": RecommendationEngine(),
        }

        self.graph = self._build_enhanced_graph()

    def _build_enhanced_graph(self):
        """Build enhanced workflow with search and vector storage"""
        workflow = StateGraph(EnhancedRFPState)

        # Add nodes
        workflow.add_node("parse_input", self._parse_input_node)
        workflow.add_node("conduct_market_research", self._market_research_node)
        workflow.add_node("manage_knowledge", self._knowledge_management_node)
        workflow.add_node("generate_suggestions", self._suggestion_node)
        workflow.add_node("enrich_security", self._security_node)
        workflow.add_node("estimate_budget", self._enhanced_budget_node)
        workflow.add_node("generate_tech_recommendations", self._tech_node)
        workflow.add_node("aggregate_requirements", self._aggregator_node)
        workflow.add_node("generate_rfp", self._rfp_generator_node)
        workflow.add_node("vendor_intelligence_agent", self._vendor_intelligence_node)
        workflow.add_node("evaluate_risks", self._risk_evaluator_node)
        workflow.add_node("generate_recommendations", self._recommendation_node)

        # Enhanced workflow edges
        workflow.add_edge(START, "parse_input")
        workflow.add_edge("parse_input", "conduct_market_research")
        workflow.add_edge("conduct_market_research", "manage_knowledge")
        workflow.add_edge("manage_knowledge", "generate_suggestions")
        workflow.add_edge("generate_suggestions", "enrich_security")
        workflow.add_edge("enrich_security", "estimate_budget")
        workflow.add_edge("estimate_budget", "generate_tech_recommendations")
        workflow.add_edge("generate_tech_recommendations", "aggregate_requirements")
        workflow.add_edge("aggregate_requirements", "generate_rfp")
        workflow.add_edge("generate_rfp", "vendor_intelligence_agent")
        workflow.add_edge("vendor_intelligence_agent", "evaluate_risks")
        workflow.add_edge("evaluate_risks", "generate_recommendations")
        workflow.add_edge("generate_recommendations", END)

        return workflow.compile()

    # Enhanced node wrapper functions with logging
    def _parse_input_node(self, state: EnhancedRFPState) -> EnhancedRFPState:
        logger.info("Starting: parse_input")
        result = self.agents["nlp_parser"].process(state)
        logger.info("Finished: parse_input")
        logger.info(f"parse_input result: {result}")
        return result

    def _market_research_node(self, state: EnhancedRFPState) -> EnhancedRFPState:
        logger.info("Starting: conduct_market_research")
        result = self.agents["market_research"].process(state)
        logger.info("Finished: conduct_market_research")
        logger.info(f"conduct_market_research result: {result}")
        return result

    def _knowledge_management_node(self, state: EnhancedRFPState) -> EnhancedRFPState:
        logger.info("Starting: manage_knowledge")
        result = self.agents["knowledge_management"].process(state)
        logger.info("Finished: manage_knowledge")
        logger.info(f"manage_knowledge result: {result}")
        return result

    def _suggestion_node(self, state: EnhancedRFPState) -> EnhancedRFPState:
        logger.info("Starting: generate_suggestions")
        result = self.agents["suggestion"].process(state)
        logger.info("Finished: generate_suggestions")
        logger.info(f"generate_suggestions result: {result}")
        return result

    def _security_node(self, state: EnhancedRFPState) -> EnhancedRFPState:
        logger.info("Starting: enrich_security")
        result = self.agents["security"].process(state)
        logger.info("Finished: enrich_security")
        logger.info(f"enrich_security result: {result}")
        return result

    def _enhanced_budget_node(self, state: EnhancedRFPState) -> EnhancedRFPState:
        logger.info("Starting: estimate_budget")
        result = self.agents["budget"].process(state)
        logger.info("Finished: estimate_budget")
        logger.info(f"estimate_budget result: {result}")
        return result

    def _tech_node(self, state: EnhancedRFPState) -> EnhancedRFPState:
        logger.info("Starting: generate_tech_recommendations")
        result = self.agents["tech"].process(state)
        logger.info("Finished: generate_tech_recommendations")
        logger.info(f"generate_tech_recommendations result: {result}")
        return result

    def _aggregator_node(self, state: EnhancedRFPState) -> EnhancedRFPState:
        logger.info("Starting: aggregate_requirements")
        result = self.agents["aggregator"].process(state)
        logger.info("Finished: aggregate_requirements")
        logger.info(f"aggregate_requirements result: {result}")
        return result

    def _rfp_generator_node(self, state: EnhancedRFPState) -> EnhancedRFPState:
        logger.info("Starting: generate_rfp")
        result = self.agents["rfp_generator"].process(state)
        logger.info("Finished: generate_rfp")
        logger.info(f"generate_rfp result: {result}")
        return result

    def _vendor_intelligence_node(self, state: EnhancedRFPState) -> EnhancedRFPState:
        logger.info("Starting: vendor_intelligence_agent")
        result = self.agents["vendor_intelligence"].process(state)
        logger.info("Finished: vendor_intelligence_agent")
        logger.info(f"vendor_intelligence_agent result: {result}")
        return result

    def _risk_evaluator_node(self, state: EnhancedRFPState) -> EnhancedRFPState:
        logger.info("Starting: evaluate_risks")
        result = self.agents["risk_evaluator"].process(state)
        logger.info("Finished: evaluate_risks")
        logger.info(f"evaluate_risks result: {result}")
        return result

    def _recommendation_node(self, state: EnhancedRFPState) -> EnhancedRFPState:
        logger.info("Starting: generate_recommendations")
        result = self.agents["recommendation"].process(state)
        logger.info("Finished: generate_recommendations")
        logger.info(f"generate_recommendations result: {result}")
        return result

    def run(self, user_input: str) -> EnhancedRFPState:
        """Execute the complete workflow"""
        logger.info("Workflow started")
        initial_state: EnhancedRFPState = {
            "user_approved": False,
            "user_input": user_input,
            "audit_log": [],
            "reviewer_comments": [],
            "analytics": {},
            "parsed_requirements": {},
            "market_research": {},
            "vendor_intelligence": {},
            "cached_knowledge": [],
            "suggestions": [],
            "security_requirements": {},
            "budget_estimate": {},
            "tech_recommendations": {},
            "aggregated_requirements": {},
            "rfp_document": "",
            "vendor_proposals": [],
            "proposal_scores": [],
            "recommendations": {},
            "visual_summary": {},
            "search_metadata": {},
        }

        result = self.graph.invoke(initial_state)
        logger.info("Workflow completed")
        return result  # type:ignore
