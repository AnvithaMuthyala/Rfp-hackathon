import streamlit as st
import time
import random
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
# ADD THESE NEW IMPORTS HERE
import openai
from openai import AzureOpenAI
import PyPDF2
import docx
import json
import io
from typing import Dict, List, Optional
import re
from dotenv import load_dotenv
# Add these LangGraph imports
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import AzureChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from typing import TypedDict, List, Annotated, Sequence
import operator
from pathlib import Path
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Proposal Generator Agent Workflow",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Company Configuration - Add this before other code
COMPANY_PROFILE = {
    "name": "TechVision Solutions",
    "industry": "Custom Software Development & Digital Solutions",
    "founded": "2015",
    "employees": "150+ professionals",
    "headquarters": "San Francisco, CA with offices in Austin, TX and London, UK",
    "specializations": [
        "Enterprise Software Development",
        "Cloud Solutions & Migration",
        "AI/ML Integration",
        "Mobile Application Development",
        "DevOps & Infrastructure",
        "Cybersecurity Solutions"
    ],
    "certifications": [
        "ISO 27001 (Information Security)",
        "SOC 2 Type II Compliance",
        "AWS Advanced Consulting Partner",
        "Microsoft Gold Partner",
        "GDPR Compliant"
    ],
    "portfolio_highlights": [
        "200+ successful projects delivered",
        "Fortune 500 clients across multiple industries",
        "99.5% client retention rate",
        "Average project delivery 15% faster than industry standard"
    ],
    "key_differentiators": [
        "Agile methodology with weekly client demos",
        "24/7 global support coverage",
        "Proprietary AI-assisted development frameworks",
        "End-to-end solution delivery",
        "Post-launch maintenance and scaling support"
    ],
    "recent_achievements": [
        "Winner of 'Best Custom Software Developer 2024' - Tech Excellence Awards",
        "Successfully migrated 50+ legacy systems to cloud",
        "Developed AI solutions saving clients avg 40% operational costs",
        "Achieved 99.9% uptime across all client deployments in 2024"
    ]
}

def get_company_context() -> str:
    """Generate company context for LLM prompts"""
    return f"""
COMPANY CONTEXT:
You are creating this proposal on behalf of {COMPANY_PROFILE['name']}, a leading {COMPANY_PROFILE['industry']} company.

Company Overview:
- Founded: {COMPANY_PROFILE['founded']}
- Team: {COMPANY_PROFILE['employees']}
- Locations: {COMPANY_PROFILE['headquarters']}

Core Specializations:
{chr(10).join([f"• {spec}" for spec in COMPANY_PROFILE['specializations']])}

Key Certifications & Compliance:
{chr(10).join([f"• {cert}" for cert in COMPANY_PROFILE['certifications']])}

Portfolio Highlights:
{chr(10).join([f"• {highlight}" for highlight in COMPANY_PROFILE['portfolio_highlights']])}

Competitive Advantages:
{chr(10).join([f"• {diff}" for diff in COMPANY_PROFILE['key_differentiators']])}

Recent Achievements:
{chr(10).join([f"• {achievement}" for achievement in COMPANY_PROFILE['recent_achievements']])}

IMPORTANT: Always write the proposal from {COMPANY_PROFILE['name']}'s perspective, highlighting our capabilities, experience, and value propositions.
"""

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .active-agent {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        animation: pulse 2s infinite;
    }
    .working-agent {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        animation: glow 1.5s ease-in-out infinite alternate;
    }
    .completed-agent {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    }
    .feedback-pending {
        background: linear-gradient(135deg, #ffa726 0%, #ff7043 100%);
        animation: attention 2s ease-in-out infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(240, 147, 251, 0.7); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 10px rgba(240, 147, 251, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(240, 147, 251, 0); }
    }
    @keyframes glow {
        from { box-shadow: 0 0 20px #4facfe; }
        to { box-shadow: 0 0 30px #00f2fe, 0 0 40px #00f2fe; }
    }
    @keyframes attention {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.03); }
    }
    .parsing-animation {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4);
        background-size: 400% 400%;
        animation: gradient 2s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-active { background-color: #ff4757; animation: blink 1s infinite; }
    .status-working { background-color: #3742fa; animation: spin 2s linear infinite; }
    .status-completed { background-color: #2ed573; }
    .status-pending { background-color: #a4b0be; }
    .status-feedback { background-color: #ffa726; animation: blink 1s infinite; }
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.3; }
    }
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    .tutorial-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    .feedback-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffa726;
        margin: 1rem 0;
    }
    .company-info {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ADD ALL AZURE OPENAI FUNCTIONS HERE (from the artifact)
# Azure OpenAI Configuration
class AzureOpenAIConfig:
    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") 
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        
    def get_client(self):
        if not self.api_key or not self.endpoint:
            return None
        
        return AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.endpoint
        )

# State definition (UPDATED - removed final orchestrator)
class ProposalState(TypedDict):
    rfp_data: dict
    current_agent: str
    agent_outputs: dict
    human_feedback: dict
    feedback_requests: List[str]
    completed_agents: List[str]
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_action: str

# Simplified LangGraph System without SQLite persistence (UPDATED)
class SimpleLangGraphProposalSystem:
    def __init__(self, config: AzureOpenAIConfig):
        self.config = config
        self.llm = self._setup_llm()
        self.workflow = self._create_workflow()
        
    def _setup_llm(self):
        """Setup Azure OpenAI LLM for LangChain"""
        if not self.config.api_key or not self.config.endpoint:
            return None
            
        return AzureChatOpenAI(
            azure_endpoint=self.config.endpoint,
            api_key=self.config.api_key,
            api_version=self.config.api_version,
            azure_deployment=self.config.deployment_name,
            temperature=0.3,
            max_tokens=2500
        )
    
    def _create_workflow(self):
        """Create the LangGraph workflow - UPDATED workflow without final orchestrator"""
        workflow = StateGraph(ProposalState)
        
        # Add nodes for each agent (REMOVED final_orchestrator)
        workflow.add_node("orchestrator", self.orchestrator_agent)
        workflow.add_node("tech_lead", self.tech_lead_agent)
        workflow.add_node("estimation", self.estimation_agent)
        workflow.add_node("timeline", self.timeline_agent)
        workflow.add_node("legal", self.legal_agent)
        workflow.add_node("sales", self.sales_agent)
        
        # Set entry point
        workflow.set_entry_point("orchestrator")
        
        # Simple linear workflow (UPDATED - sales goes to END)
        workflow.add_edge("orchestrator", "tech_lead")
        workflow.add_edge("tech_lead", "estimation")
        workflow.add_edge("estimation", "timeline")
        workflow.add_edge("timeline", "legal")
        workflow.add_edge("legal", "sales")
        workflow.add_edge("sales", END)  # CHANGED: sales now goes directly to END
        
        # Compile without checkpointer
        return workflow.compile()
    
    def orchestrator_agent(self, state: ProposalState) -> ProposalState:
        """First agent - Proposal Orchestrator"""
        if not self.llm:
            output = self._get_mock_output("Proposal Orchestrator Agent")
        else:
            prompt = self._create_orchestrator_prompt(state["rfp_data"])
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            output = response.content
        
        # Update state
        state["agent_outputs"]["Proposal Orchestrator Agent"] = output
        state["completed_agents"].append("Proposal Orchestrator Agent")
        state["current_agent"] = "Tech Lead Agent"
        
        return state
    
    def tech_lead_agent(self, state: ProposalState) -> ProposalState:
        """Tech Lead Agent"""
        if not self.llm:
            output = self._get_mock_output("Tech Lead Agent")
        else:
            prompt = self._create_tech_lead_prompt(state["rfp_data"], state.get("human_feedback", {}))
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            output = response.content
        
        state["agent_outputs"]["Tech Lead Agent"] = output
        state["completed_agents"].append("Tech Lead Agent")
        state["current_agent"] = "Estimation Agent"
        
        return state
    
    def estimation_agent(self, state: ProposalState) -> ProposalState:
        """Estimation Agent"""
        if not self.llm:
            output = self._get_mock_output("Estimation Agent")
        else:
            prompt = self._create_estimation_prompt(state["rfp_data"], state.get("human_feedback", {}))
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            output = response.content
        
        state["agent_outputs"]["Estimation Agent"] = output
        state["completed_agents"].append("Estimation Agent")
        state["current_agent"] = "Timeline Agent"
        
        return state
    
    def timeline_agent(self, state: ProposalState) -> ProposalState:
        """Timeline Agent"""
        if not self.llm:
            output = self._get_mock_output("Timeline Agent")
        else:
            prompt = self._create_timeline_prompt(state["rfp_data"], state.get("human_feedback", {}))
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            output = response.content
        
        state["agent_outputs"]["Timeline Agent"] = output
        state["completed_agents"].append("Timeline Agent")
        state["current_agent"] = "Legal & Compliance Agent"
        
        return state
    
    def legal_agent(self, state: ProposalState) -> ProposalState:
        """Legal & Compliance Agent"""
        if not self.llm:
            output = self._get_mock_output("Legal & Compliance Agent")
        else:
            prompt = self._create_legal_prompt(state["rfp_data"], state.get("human_feedback", {}))
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            output = response.content
        
        state["agent_outputs"]["Legal & Compliance Agent"] = output
        state["completed_agents"].append("Legal & Compliance Agent")
        state["current_agent"] = "Sales/Marketing Agent"
        
        return state
    
    def sales_agent(self, state: ProposalState) -> ProposalState:
        """Sales/Marketing Agent - UPDATED to be the final agent"""
        if not self.llm:
            output = self._get_mock_output("Sales/Marketing Agent")
        else:
            prompt = self._create_sales_prompt(state["rfp_data"], state.get("human_feedback", {}))
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            output = response.content
        
        state["agent_outputs"]["Sales/Marketing Agent"] = output
        state["completed_agents"].append("Sales/Marketing Agent")
        state["current_agent"] = "completed"  # CHANGED: mark as completed instead of going to final orchestrator
        
        return state
    
    # All the prompt creation methods (UPDATED with company context)
    def _create_orchestrator_prompt(self, rfp_data: dict) -> str:
        company_context = get_company_context()
        return f"""{company_context}

You are a Proposal Orchestrator Agent working for {COMPANY_PROFILE['name']}. Based on the RFP analysis below, create a comprehensive project breakdown.

RFP Data: {rfp_data}

Create a detailed response covering:
1. Project scope overview and understanding from {COMPANY_PROFILE['name']}'s perspective
2. Key components identification and prioritization based on our expertise
3. Risk assessment and mitigation strategies leveraging our experience
4. Success criteria and metrics aligned with our proven methodologies

Format as a professional proposal section with clear headings. Emphasize {COMPANY_PROFILE['name']}'s relevant experience and capabilities."""
    
    def _create_tech_lead_prompt(self, rfp_data: dict, feedback: dict) -> str:
        company_context = get_company_context()
        feedback_text = feedback.get("Tech Lead Agent", "")
        feedback_context = f"\n\nHuman Feedback: {feedback_text}" if feedback_text else ""
        
        return f"""{company_context}

You are a Technical Lead Agent representing {COMPANY_PROFILE['name']}. Design the technical architecture based on our proven expertise.

RFP Data: {rfp_data}
{feedback_context}

Provide:
1. Recommended technology stack leveraging {COMPANY_PROFILE['name']}'s specializations
2. System architecture design based on our enterprise experience
3. Security implementation strategy aligned with our ISO 27001 and SOC 2 certifications
4. Development methodology using our proven Agile processes

Be specific about technologies and highlight how {COMPANY_PROFILE['name']}'s expertise ensures successful implementation."""
    
    def _create_estimation_prompt(self, rfp_data: dict, feedback: dict) -> str:
        company_context = get_company_context()
        feedback_text = feedback.get("Estimation Agent", "")
        feedback_context = f"\n\nHuman Feedback: {feedback_text}" if feedback_text else ""
        
        return f"""{company_context}

You are an Estimation Agent for {COMPANY_PROFILE['name']}. Provide detailed cost estimates based on our proven delivery track record.

RFP Data: {rfp_data}
{feedback_context}

Provide:
1. Work breakdown structure based on {COMPANY_PROFILE['name']}'s proven methodologies
2. Effort estimation by component using our historical data and expertise
3. Resource allocation leveraging our {COMPANY_PROFILE['employees']} team
4. Total project cost with breakdown, highlighting our competitive advantage of 15% faster delivery

Reference our portfolio of 200+ successful projects for credibility."""
    
    def _create_timeline_prompt(self, rfp_data: dict, feedback: dict) -> str:
        company_context = get_company_context()
        feedback_text = feedback.get("Timeline Agent", "")
        feedback_context = f"\n\nHuman Feedback: {feedback_text}" if feedback_text else ""
        
        return f"""{company_context}

You are a Timeline Agent for {COMPANY_PROFILE['name']}. Create project timeline based on our proven delivery methodologies.

RFP Data: {rfp_data}
{feedback_context}

Develop:
1. Project phases and milestones using {COMPANY_PROFILE['name']}'s Agile methodology
2. Task breakdown and dependencies based on our proven processes
3. Resource scheduling leveraging our global team capabilities
4. Delivery schedule with key dates, emphasizing our track record of 15% faster delivery

Highlight our weekly client demos and 24/7 global support coverage."""
    
    def _create_legal_prompt(self, rfp_data: dict, feedback: dict) -> str:
        company_context = get_company_context()
        feedback_text = feedback.get("Legal & Compliance Agent", "")
        feedback_context = f"\n\nHuman Feedback: {feedback_text}" if feedback_text else ""
        
        return f"""{company_context}

You are a Legal & Compliance Agent for {COMPANY_PROFILE['name']}. Provide legal guidance based on our certifications and compliance expertise.

RFP Data: {rfp_data}
{feedback_context}

Address:
1. Regulatory compliance requirements leveraging our ISO 27001 and SOC 2 Type II certifications
2. Data protection considerations based on our GDPR compliance expertise
3. Contract terms recommendations from our experience with Fortune 500 clients
4. Risk assessment using insights from our 200+ successful projects

Emphasize {COMPANY_PROFILE['name']}'s proven compliance track record and industry certifications."""
    
    def _create_sales_prompt(self, rfp_data: dict, feedback: dict) -> str:
        company_context = get_company_context()
        feedback_text = feedback.get("Sales/Marketing Agent", "")
        feedback_context = f"\n\nHuman Feedback: {feedback_text}" if feedback_text else ""
        
        return f"""{company_context}

You are a Sales/Marketing Agent for {COMPANY_PROFILE['name']}. Create compelling value propositions highlighting our competitive advantages.

RFP Data: {rfp_data}
{feedback_context}

Develop:
1. Executive summary showcasing {COMPANY_PROFILE['name']}'s unique value proposition
2. Company capabilities highlighting our specializations and achievements
3. Competitive advantages including our 99.5% client retention rate and recent awards
4. Client benefits and ROI based on our track record of saving clients 40% operational costs

Create a compelling closing that positions {COMPANY_PROFILE['name']} as the ideal partner for this project."""
    
    def _get_mock_output(self, agent_name: str) -> str:
        """Mock outputs when Azure OpenAI isn't configured - UPDATED with company context"""
        mock_outputs = {
            "Proposal Orchestrator Agent": f"**Project Analysis Complete** - {COMPANY_PROFILE['name']} has successfully analyzed RFP requirements and identified 6 key project components based on our expertise in {', '.join(COMPANY_PROFILE['specializations'][:3])}.",
            "Tech Lead Agent": f"**Technical Architecture Designed** - {COMPANY_PROFILE['name']} recommends modern tech stack with React frontend, Node.js backend, AWS deployment leveraging our AWS Advanced Consulting Partner status.",
            "Estimation Agent": f"**Cost Analysis Complete** - Based on {COMPANY_PROFILE['name']}'s 200+ successful projects, total estimate: $85,000 over 26 weeks with detailed component breakdown and 15% faster delivery.",
            "Timeline Agent": f"**Project Timeline Created** - {COMPANY_PROFILE['name']}'s proven Agile methodology: 26-week schedule with weekly client demos and 24/7 support coverage.",
            "Legal & Compliance Agent": f"**Compliance Review Complete** - {COMPANY_PROFILE['name']}'s ISO 27001 and SOC 2 Type II certifications ensure GDPR compliance with low risk assessment.",
            "Sales/Marketing Agent": f"**Value Proposition Developed** - {COMPANY_PROFILE['name']}'s competitive advantages: 99.5% client retention, 200+ successful projects, and 2024 'Best Custom Software Developer' award winner."
        }
        return mock_outputs.get(agent_name, f"Mock output for {agent_name}")
    
    def run_single_agent(self, agent_name: str, state: ProposalState) -> ProposalState:
        """Run a single agent - UPDATED agent list"""
        agent_functions = {
            "Proposal Orchestrator Agent": self.orchestrator_agent,
            "Tech Lead Agent": self.tech_lead_agent,
            "Estimation Agent": self.estimation_agent,
            "Timeline Agent": self.timeline_agent,
            "Legal & Compliance Agent": self.legal_agent,
            "Sales/Marketing Agent": self.sales_agent
            # REMOVED: "Proposal Orchestrator Agent (Final)"
        }
        
        if agent_name in agent_functions:
            return agent_functions[agent_name](state)
        return state

    def generate_final_proposal(self, state: ProposalState) -> str:
        """NEW: Generate final consolidated proposal from all agent outputs"""
        if not state["agent_outputs"]:
            return "No agent outputs available to consolidate."
        
        # Create header with company information
        header = f"""
# Proposal Response to RFP

**Submitted by:** {COMPANY_PROFILE['name']}
**Date:** {datetime.now().strftime('%B %d, %Y')}
**Company:** {COMPANY_PROFILE['industry']}
**Headquarters:** {COMPANY_PROFILE['headquarters']}

---

## Executive Summary

{COMPANY_PROFILE['name']} is pleased to submit this comprehensive proposal in response to your RFP. With over {len(COMPANY_PROFILE['portfolio_highlights'])} years of experience and {COMPANY_PROFILE['portfolio_highlights'][0]}, we are uniquely positioned to deliver exceptional results for your project.

---
"""
        
        # Combine all agent outputs in logical order
        agent_order = [
            "Proposal Orchestrator Agent",
            "Tech Lead Agent", 
            "Estimation Agent",
            "Timeline Agent",
            "Legal & Compliance Agent",
            "Sales/Marketing Agent"
        ]
        
        proposal_sections = []
        for agent_name in agent_order:
            if agent_name in state["agent_outputs"]:
                section_title = agent_name.replace(" Agent", "").replace("Proposal Orchestrator", "Project Overview")
                proposal_sections.append(f"## {section_title}\n\n{state['agent_outputs'][agent_name]}\n")
        
        # Add company footer
        footer = f"""
---

## About {COMPANY_PROFILE['name']}

**Why Choose {COMPANY_PROFILE['name']}:**
{chr(10).join([f"• {diff}" for diff in COMPANY_PROFILE['key_differentiators']])}

**Recent Achievements:**
{chr(10).join([f"• {achievement}" for achievement in COMPANY_PROFILE['recent_achievements']])}

**Contact Information:**
- Company: {COMPANY_PROFILE['name']}
- Industry: {COMPANY_PROFILE['industry']}
- Locations: {COMPANY_PROFILE['headquarters']}
- Team Size: {COMPANY_PROFILE['employees']}

We look forward to partnering with you on this exciting project.

---
*This proposal was generated by {COMPANY_PROFILE['name']}'s AI-assisted proposal system, ensuring comprehensive coverage while maintaining our personal touch and expertise.*
"""
        
        return header + "\n".join(proposal_sections) + footer

# Simplified system getter
def get_simple_langgraph_system():
    """Get or create simplified LangGraph system"""
    if 'langgraph_system' not in st.session_state:
        config = AzureOpenAIConfig()
        st.session_state.langgraph_system = SimpleLangGraphProposalSystem(config)
    return st.session_state.langgraph_system

def render_manual_langgraph_ui():
    """Manual control version - UPDATED to remove final orchestrator"""
    st.title("🤖 Agent Workflow Grid")
    
    # Add company info banner
    st.markdown(f"""
    <div class="company-info">
        <h3>🏢 Proposal by {COMPANY_PROFILE['name']}</h3>
        <p><strong>Industry:</strong> {COMPANY_PROFILE['industry']} | <strong>Team:</strong> {COMPANY_PROFILE['employees']} | <strong>Founded:</strong> {COMPANY_PROFILE['founded']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("Watch as specialized AI agents work on different aspects of your proposal")
    
    # Check parsed data
    if 'parsed_rfp_data' not in st.session_state:
        st.error("❌ No parsed RFP data available.")
        if st.button("⬅️ Back to Parsing"):
            st.session_state.step = 'parsing'
            st.rerun()
        return
    
    # Initialize LangGraph system
    langgraph_system = get_simple_langgraph_system()
    
    # Initialize workflow state
    if 'workflow_state' not in st.session_state:
        st.session_state.workflow_state = ProposalState(
            rfp_data=st.session_state.parsed_rfp_data,
            current_agent="Proposal Orchestrator Agent",
            agent_outputs={},
            human_feedback={},
            feedback_requests=[],
            completed_agents=[],
            messages=[],
            next_action="start"
        )
    
    # Agent status summary - UPDATED count
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        active_count = 1 if st.session_state.workflow_state["current_agent"] != "completed" else 0
        st.metric("Active Agents", active_count)
    with col2:
        working_count = 0  # Manual control, so no working state
        st.metric("Working Agents", working_count)
    with col3:
        completed_count = len(st.session_state.workflow_state["completed_agents"])
        st.metric("Completed", completed_count)
    with col4:
        progress = (completed_count / 6) * 100  # CHANGED: now 6 agents instead of 7
        st.metric("Overall Progress", f"{progress:.1f}%")
    
    st.markdown("---")
    
    # Current agent info
    current_agent = st.session_state.workflow_state["current_agent"]
    
    if current_agent != "completed":
        st.info(f"🎯 **Next Agent:** {current_agent}")
    else:
        st.success("🎉 **All Agents Completed!**")
    
    # Control buttons - UPDATED logic
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Handle quick run from agent cards
        if hasattr(st.session_state, 'quick_run_agent') and st.session_state.quick_run_agent == current_agent:
            del st.session_state.quick_run_agent
            # Run the agent immediately
            try:
                st.session_state.workflow_state = langgraph_system.run_single_agent(
                    current_agent, 
                    st.session_state.workflow_state
                )
                
                # Update UI state
                st.session_state.agents_workflow[current_agent]["status"] = "completed"
                st.session_state.agents_workflow[current_agent]["progress"] = 100
                
                # Store the output
                if current_agent in st.session_state.workflow_state["agent_outputs"]:
                    st.session_state.agents_workflow[current_agent]["output"] = st.session_state.workflow_state["agent_outputs"][current_agent]
                
                # Request feedback after completion
                st.session_state.agents_workflow[current_agent]["feedback_requested"] = True
                st.session_state.feedback_target_agent = current_agent
                
                st.success(f"✅ {current_agent} completed!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error running {current_agent}: {str(e)}")
        
        if current_agent != "completed":
            if st.button(f"🚀 Run {current_agent}", type="primary", key="run_current"):
                # Run the current agent
                try:
                    st.session_state.workflow_state = langgraph_system.run_single_agent(
                        current_agent, 
                        st.session_state.workflow_state
                    )
                    
                    # Update UI state
                    st.session_state.agents_workflow[current_agent]["status"] = "completed"
                    st.session_state.agents_workflow[current_agent]["progress"] = 100
                    
                    # Store the output
                    if current_agent in st.session_state.workflow_state["agent_outputs"]:
                        st.session_state.agents_workflow[current_agent]["output"] = st.session_state.workflow_state["agent_outputs"][current_agent]
                    
                    # Request feedback after completion
                    st.session_state.agents_workflow[current_agent]["feedback_requested"] = True
                    st.session_state.feedback_target_agent = current_agent
                    
                    st.success(f"✅ {current_agent} completed!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error running {current_agent}: {str(e)}")
        else:
            st.success("🎉 All agents completed!")
    
    with col2:
        if current_agent != "completed":
            if st.button("⚡ Run All Remaining", key="run_all"):
                # Get list of agents to run - UPDATED agent list
                remaining_agents = []
                temp_agent = current_agent
                
                # Simple mapping of agent sequence - REMOVED final orchestrator
                agent_sequence = [
                    "Proposal Orchestrator Agent",
                    "Tech Lead Agent", 
                    "Estimation Agent",
                    "Timeline Agent",
                    "Legal & Compliance Agent",
                    "Sales/Marketing Agent"
                ]
                
                # Find remaining agents
                current_index = agent_sequence.index(temp_agent) if temp_agent in agent_sequence else 0
                remaining_agents = agent_sequence[current_index:]
                
                # Run all remaining agents with a single progress indicator
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_agents = len(remaining_agents)
                
                for i, agent in enumerate(remaining_agents):
                    if agent not in st.session_state.workflow_state["completed_agents"]:
                        # Update progress (convert to 0.0-1.0 range)
                        progress = i / total_agents
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {agent}... ({i+1}/{total_agents})")
                        
                        try:
                            st.session_state.workflow_state = langgraph_system.run_single_agent(
                                agent, 
                                st.session_state.workflow_state
                            )
                            
                            # Update UI
                            st.session_state.agents_workflow[agent]["status"] = "completed"
                            st.session_state.agents_workflow[agent]["progress"] = 100
                            
                            if agent in st.session_state.workflow_state["agent_outputs"]:
                                st.session_state.agents_workflow[agent]["output"] = st.session_state.workflow_state["agent_outputs"][agent]
                            
                            # For batch processing, don't request individual feedback
                            # Just mark as completed
                            
                        except Exception as e:
                            st.error(f"Error running {agent}: {str(e)}")
                            break
                
                # Complete progress (ensure it's 1.0, not 100)
                progress_bar.progress(1.0)
                status_text.text("✅ All agents completed!")
                time.sleep(1)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.success("🎉 All remaining agents completed!")
                st.balloons()
                st.rerun()
    
    with col3:
        if st.button("🔄 Reset All", key="reset_all"):
            # Reset everything
            for agent_info in st.session_state.agents_workflow.values():
                agent_info["status"] = "pending"
                agent_info["progress"] = 0
                agent_info["output"] = ""
                agent_info["feedback_requested"] = False
                agent_info["feedback_incorporated"] = False
            
            st.session_state.workflow_state = ProposalState(
                rfp_data=st.session_state.parsed_rfp_data,
                current_agent="Proposal Orchestrator Agent",
                agent_outputs={},
                human_feedback={},
                feedback_requests=[],
                completed_agents=[],
                messages=[],
                next_action="start"
            )
            
            st.session_state.agents_workflow["Proposal Orchestrator Agent"]["status"] = "active"
            st.rerun()
    
    # Debug info (helpful for troubleshooting)
    with st.expander("🔍 Debug Info"):
        st.write(f"**Current Agent:** {current_agent}")
        st.write(f"**Completed Agents:** {st.session_state.workflow_state['completed_agents']}")
        st.write(f"**Available Outputs:** {list(st.session_state.workflow_state['agent_outputs'].keys())}")
    
    # Your beautiful agent grid display - UPDATED to exclude final orchestrator
    st.markdown("### 🤖 Agent Status Grid")
    cols = st.columns(3)
    
    for i, (agent_name, agent_info) in enumerate(st.session_state.agents_workflow.items()):
        # Skip the final orchestrator agent if it exists in the workflow
        if "Final" in agent_name:
            continue
            
        with cols[i % 3]:
            # Determine status and styling
            if agent_name in st.session_state.workflow_state["completed_agents"]:
                agent_info["status"] = "completed"
                agent_info["progress"] = 100
                css_class = "agent-card completed-agent"
                status_indicator = '<span class="status-indicator status-completed"></span>'
            elif agent_name == current_agent and current_agent != "completed":
                agent_info["status"] = "active"
                css_class = "agent-card active-agent"
                status_indicator = '<span class="status-indicator status-active"></span>'
            else:
                agent_info["status"] = "pending"
                css_class = "agent-card"
                status_indicator = '<span class="status-indicator status-pending"></span>'
            
            # Your beautiful card display
            st.markdown(f"""
            <div class="{css_class}">
                <h4>{status_indicator}{agent_name}</h4>
                <p><strong>Task:</strong> {agent_info['task']}</p>
                <p><strong>Status:</strong> {agent_info['status'].capitalize()}</p>
                <p><strong>Progress:</strong> {agent_info['progress']}%</p>
                <p><strong>ETA:</strong> {agent_info['estimated_time']}</p>
                {f"<p><strong>🔔 Feedback Requested</strong></p>" if agent_info['feedback_requested'] and not agent_info['feedback_incorporated'] else ""}
                {f"<p><strong>✅ Feedback Received</strong></p>" if agent_info.get('feedback_incorporated') else ""}
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar (ensure values are between 0.0 and 1.0)
            if agent_info["status"] == "active":
                st.progress(0.0, text=f"{agent_name} - Ready to run")
            elif agent_info["status"] == "completed":
                st.progress(1.0, text="✅ Completed")
            else:
                st.progress(0.0, text="⏳ Waiting")
            
            # Show better output preview if completed - using modal
            if agent_info["status"] == "completed" and agent_info.get("output"):
                if st.button(f"👁️ View Output", key=f"view_{i}", help=f"View {agent_name} output"):
                    st.session_state.modal_agent = agent_name
                    st.session_state.modal_type = "preview"
                    st.rerun()
            
            # Show what will be generated for pending agents
            elif agent_info["status"] == "pending":
                if st.button(f"ℹ️ Info", key=f"info_{i}", help=f"See what {agent_name} will generate"):
                    st.session_state.modal_agent = agent_name
                    st.session_state.modal_type = "info"
                    st.rerun()
            
            # Show active status for current agent
            elif agent_info["status"] == "active":
                st.info(f"🎯 Ready to generate content")
                if st.button(f"▶️ Quick Run", key=f"quick_{i}", help=f"Run {agent_name} now"):
                    # Trigger the main run button
                    st.session_state.quick_run_agent = agent_name
                    st.rerun()
    
    # Check for feedback requests - define feedback_pending variable
    feedback_pending = any(agent["feedback_requested"] and not agent["feedback_incorporated"] 
                          for agent in st.session_state.agents_workflow.values())
    
    # Modal/Popup for agent output preview and feedback
    if hasattr(st.session_state, 'modal_agent') and st.session_state.modal_agent:
        modal_agent = st.session_state.modal_agent
        modal_type = getattr(st.session_state, 'modal_type', 'preview')
        agent_info = st.session_state.agents_workflow[modal_agent]
        
        # Create modal using container and styling
        with st.container():
            st.markdown("""
            <style>
            .modal-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.5);
                z-index: 1000;
            }
            .modal-content {
                background: white;
                margin: 5% auto;
                padding: 20px;
                border-radius: 10px;
                width: 80%;
                max-width: 800px;
                max-height: 80vh;
                overflow-y: auto;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Modal header
            col1, col2 = st.columns([4, 1])
            with col1:
                if modal_type == "preview":
                    st.markdown(f"### 📄 {modal_agent} - Generated Output")
                else:
                    st.markdown(f"### ℹ️ {modal_agent} - Information")
            
            with col2:
                if st.button("✖️", key="close_modal", help="Close"):
                    if hasattr(st.session_state, 'modal_agent'):
                        del st.session_state.modal_agent
                    if hasattr(st.session_state, 'modal_type'):
                        del st.session_state.modal_type
                    st.rerun()
            
            st.markdown("---")
            
            if modal_type == "preview" and agent_info.get("output"):
                # Show the agent output
                output_text = agent_info["output"]
                
                # Format output nicely
                if "**" in output_text:  # Already has markdown formatting
                    st.markdown(output_text)
                else:
                    # Add basic structure if plain text
                    lines = output_text.split('\n')
                    formatted_lines = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('-') and not line.startswith('•'):
                            if len(line) < 100 and line.endswith(':'):
                                formatted_lines.append(f"### {line}")
                            else:
                                formatted_lines.append(line)
                        elif line:
                            formatted_lines.append(line)
                    
                    formatted_output = '\n\n'.join(formatted_lines)
                    st.markdown(formatted_output)
                
                # Show statistics
                word_count = len(output_text.split())
                char_count = len(output_text)
                st.caption(f"📊 {word_count} words • {char_count} characters")
                
                st.markdown("---")
                
                # Feedback section in the same modal
                st.markdown("### 💬 Provide Feedback (Optional)")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    feedback_text = st.text_area(
                        "Your feedback:",
                        placeholder=f"Share your thoughts on {modal_agent}'s output. Suggest improvements, corrections, or additional requirements...",
                        height=120,
                        key=f"modal_feedback_{modal_agent}"
                    )
                    
                    feedback_type = st.selectbox(
                        "Feedback type:",
                        ["General Review", "Technical Correction", "Business Insight", "Additional Requirements", "Approval"],
                        key=f"modal_feedback_type_{modal_agent}"
                    )
                
                with col2:
                    st.markdown("**Quick Actions**")
                    
                    # Quick approval
                    if st.button("✅ Approve", type="primary", key=f"modal_approve_{modal_agent}"):
                        # Mark feedback as handled
                        agent_info["feedback_requested"] = False
                        agent_info["feedback_incorporated"] = True
                        agent_info["human_feedback"] = "Approved"
                        
                        # Store in feedback history
                        feedback_entry = {
                            "agent": modal_agent,
                            "type": "Approval",
                            "content": "Output approved by user",
                            "priority": "Medium",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.session_state.feedback_history.append(feedback_entry)
                        
                        # Close modal
                        del st.session_state.modal_agent
                        if hasattr(st.session_state, 'modal_type'):
                            del st.session_state.modal_type
                        if hasattr(st.session_state, 'feedback_target_agent'):
                            del st.session_state.feedback_target_agent
                        
                        st.success(f"✅ {modal_agent} approved!")
                        st.rerun()
                    
                    # Submit detailed feedback
                    if st.button("📤 Submit", key=f"modal_submit_{modal_agent}"):
                        if feedback_text.strip():
                            # Store detailed feedback
                            agent_info["feedback_requested"] = False
                            agent_info["feedback_incorporated"] = True
                            agent_info["human_feedback"] = feedback_text
                            
                            # Store in feedback history
                            feedback_entry = {
                                "agent": modal_agent,
                                "type": feedback_type,
                                "content": feedback_text,
                                "priority": "Medium",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            st.session_state.feedback_history.append(feedback_entry)
                            
                            # Update LangGraph state with feedback
                            if hasattr(st.session_state, 'workflow_state'):
                                if "human_feedback" not in st.session_state.workflow_state:
                                    st.session_state.workflow_state["human_feedback"] = {}
                                st.session_state.workflow_state["human_feedback"][modal_agent] = feedback_text
                            
                            # Close modal
                            del st.session_state.modal_agent
                            if hasattr(st.session_state, 'modal_type'):
                                del st.session_state.modal_type
                            if hasattr(st.session_state, 'feedback_target_agent'):
                                del st.session_state.feedback_target_agent
                            
                            st.success(f"✅ Feedback submitted!")
                            st.rerun()
                        else:
                            st.warning("Please provide feedback text or use 'Approve' for quick approval.")
                    
                    # Skip feedback
                    if st.button("⏭️ Skip", key=f"modal_skip_{modal_agent}"):
                        # Mark as handled without feedback
                        agent_info["feedback_requested"] = False
                        agent_info["feedback_incorporated"] = True
                        
                        # Close modal
                        del st.session_state.modal_agent
                        if hasattr(st.session_state, 'modal_type'):
                            del st.session_state.modal_type
                        if hasattr(st.session_state, 'feedback_target_agent'):
                            del st.session_state.feedback_target_agent
                        
                        st.rerun()
                
                # Show feedback guidelines
                with st.expander("💡 Feedback Guidelines"):
                    st.markdown("""
                    **How to provide effective feedback:**
                    
                    **✅ Helpful feedback:**
                    - Suggest specific improvements
                    - Point out missing information  
                    - Recommend alternative approaches
                    - Share domain expertise
                    
                    **❌ Less helpful:**
                    - Vague comments like "good" or "bad"
                    - Feedback outside the agent's scope
                    - Personal preferences without reasoning
                    """)
            
            elif modal_type == "info":
                # Show agent information - UPDATED descriptions with company context
                agent_descriptions = {
                    "Proposal Orchestrator Agent": f"Project scope overview, risk assessment, success criteria, and stakeholder management plan based on {COMPANY_PROFILE['name']}'s proven methodologies",
                    "Tech Lead Agent": f"Technology stack recommendations, system architecture, security strategy leveraging {COMPANY_PROFILE['name']}'s certifications and development methodology", 
                    "Estimation Agent": f"Detailed cost breakdown, effort estimation, resource allocation based on {COMPANY_PROFILE['name']}'s 200+ project experience",
                    "Timeline Agent": f"Project phases, milestones, task dependencies using {COMPANY_PROFILE['name']}'s Agile methodology and delivery schedule",
                    "Legal & Compliance Agent": f"Regulatory compliance analysis leveraging {COMPANY_PROFILE['name']}'s ISO 27001 and SOC 2 certifications, contract terms, and risk assessment",
                    "Sales/Marketing Agent": f"Executive summary highlighting {COMPANY_PROFILE['name']}'s value propositions, competitive advantages, and client benefits"
                }
                
                description = agent_descriptions.get(modal_agent, "Specialized proposal content")
                
                st.info(f"📝 **What this agent will generate:**\n\n{description}")
                
                st.markdown("### 🎯 Agent Responsibilities:")
                
                if modal_agent == "Proposal Orchestrator Agent":
                    st.markdown(f"""
                    - Analyze RFP requirements using {COMPANY_PROFILE['name']}'s proven framework
                    - Identify key project components based on our specializations
                    - Assess potential risks using insights from 200+ projects
                    - Define success criteria aligned with our methodologies
                    - Create stakeholder management approach
                    """)
                elif modal_agent == "Tech Lead Agent":
                    st.markdown(f"""
                    - Recommend optimal technology stack from {COMPANY_PROFILE['name']}'s expertise
                    - Design system architecture leveraging our AWS Advanced Consulting Partner status
                    - Plan security implementation using our ISO 27001 and SOC 2 certifications
                    - Define development methodology based on our proven Agile processes
                    - Specify integration points and APIs
                    """)
                elif modal_agent == "Estimation Agent":
                    st.markdown(f"""
                    - Break down work using {COMPANY_PROFILE['name']}'s proven methodologies
                    - Estimate effort based on our 200+ successful projects
                    - Calculate resource requirements from our {COMPANY_PROFILE['employees']} team
                    - Factor in risk buffers and our 15% faster delivery advantage
                    - Provide transparent cost justification
                    """)
                elif modal_agent == "Timeline Agent":
                    st.markdown(f"""
                    - Define project phases using {COMPANY_PROFILE['name']}'s Agile methodology
                    - Map task dependencies based on our proven processes
                    - Schedule resource allocation leveraging our global team
                    - Plan review gates with our weekly client demos
                    - Set realistic delivery expectations with 24/7 support coverage
                    """)
                elif modal_agent == "Legal & Compliance Agent":
                    st.markdown(f"""
                    - Review compliance requirements using {COMPANY_PROFILE['name']}'s certifications
                    - Analyze contract terms based on Fortune 500 client experience
                    - Assess legal risks using insights from 200+ projects
                    - Recommend data protection measures leveraging GDPR compliance
                    - Ensure industry-specific compliance standards
                    """)
                elif modal_agent == "Sales/Marketing Agent":
                    st.markdown(f"""
                    - Craft value propositions highlighting {COMPANY_PROFILE['name']}'s advantages
                    - Showcase our 99.5% client retention rate and recent awards
                    - Create executive summary demonstrating our expertise
                    - Demonstrate ROI based on our track record of 40% cost savings
                    - Position {COMPANY_PROFILE['name']} as the ideal partner
                    """)
                
                if st.button("🚀 Run This Agent Now", key=f"modal_run_{modal_agent}", type="primary"):
                    if modal_agent == current_agent:
                        st.session_state.quick_run_agent = modal_agent
                        del st.session_state.modal_agent
                        if hasattr(st.session_state, 'modal_type'):
                            del st.session_state.modal_type
                        st.rerun()
                    else:
                        st.warning(f"Please run agents in sequence. Current agent: {current_agent}")
    
    # Handle automatic feedback requests (when agent completes)
    elif feedback_pending and hasattr(st.session_state, 'feedback_target_agent'):
        # Automatically open modal for feedback
        st.session_state.modal_agent = st.session_state.feedback_target_agent
        st.session_state.modal_type = "preview"
        del st.session_state.feedback_target_agent
        st.rerun()
    
    elif feedback_pending:
        st.warning("🔔 One or more agents are requesting human feedback!")
        # Find the agent requesting feedback and open modal
        for agent_name, agent_info in st.session_state.agents_workflow.items():
            if agent_info["feedback_requested"] and not agent_info["feedback_incorporated"]:
                if st.button(f"💬 Review {agent_name}", type="primary"):
                    st.session_state.modal_agent = agent_name
                    st.session_state.modal_type = "preview"
                    st.rerun()
                break
    
    # Completion handling - UPDATED
    if current_agent == "completed":
        st.success("🎉 All agents have completed their work!")
        
        # Show summary
        if st.session_state.workflow_state["agent_outputs"]:
            st.markdown("### 📊 Generated Content Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                total_words = sum(len(output.split()) for output in st.session_state.workflow_state["agent_outputs"].values())
                st.metric("Total Words Generated", total_words)
            with col2:
                st.metric("Sections Created", len(st.session_state.workflow_state["agent_outputs"]))
            with col3:
                avg_words = total_words // len(st.session_state.workflow_state["agent_outputs"]) if st.session_state.workflow_state["agent_outputs"] else 0
                st.metric("Avg Words/Section", avg_words)
        
        # NEW: Generate consolidated proposal automatically
        if st.button("📋 Generate Final Proposal", type="primary"):
            # Generate the final consolidated proposal
            final_proposal = langgraph_system.generate_final_proposal(st.session_state.workflow_state)
            st.session_state.consolidated_document = final_proposal
            st.session_state.step = 'consolidate'
            st.rerun()
    
    # Azure OpenAI status
    config = AzureOpenAIConfig()
    if not config.api_key or not config.endpoint:
        st.warning("⚠️ Azure OpenAI not configured. Agents will produce mock responses.")
    else:
        st.info("✅ Azure OpenAI configured. Agents will generate real content.")

# Add this debugging function after your LangGraph classes
def debug_agent_states():
    """Debug function to see agent states"""
    if st.sidebar.button("🔍 Debug States"):
        st.sidebar.markdown("### LangGraph State")
        st.sidebar.write(f"Current: {st.session_state.workflow_state['current_agent']}")
        st.sidebar.write(f"Completed: {st.session_state.workflow_state['completed_agents']}")
        
        st.sidebar.markdown("### UI Agent States")
        for name, info in st.session_state.agents_workflow.items():
            st.sidebar.write(f"{name}: {info['status']}")

# Document Processing Functions
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(uploaded_file) -> str:
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(io.BytesIO(uploaded_file.read()))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def extract_text_from_txt(uploaded_file) -> str:
    """Extract text from TXT file"""
    try:
        return uploaded_file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
        return ""

def extract_text_from_file(uploaded_file) -> str:
    """Extract text based on file type"""
    file_type = uploaded_file.type
    
    if file_type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(uploaded_file)
    elif file_type == "text/plain":
        return extract_text_from_txt(uploaded_file)
    else:
        st.error(f"Unsupported file type: {file_type}")
        return ""

# Azure OpenAI Parsing Functions
def create_rfp_analysis_prompt(rfp_text: str) -> str:
    """Create a comprehensive prompt for RFP analysis - UPDATED with company context"""
    company_context = get_company_context()
    return f"""{company_context}

You are an expert RFP analyst working for {COMPANY_PROFILE['name']}. Analyze the following RFP document and extract key information that will help our specialized agents create a winning proposal.

RFP Document:
{rfp_text}

Please analyze this RFP from {COMPANY_PROFILE['name']}'s perspective and provide a JSON response with the following structure:

{{
    "project_overview": {{
        "title": "Project title",
        "description": "Brief project description",
        "type": "Type of project (e.g., Software Development, Infrastructure, etc.)"
    }},
    "technical_requirements": [
        "List of technical requirements that align with our specializations"
    ],
    "functional_requirements": [
        "List of functional requirements"
    ],
    "compliance_requirements": [
        "List of compliance and regulatory requirements (consider our certifications)"
    ],
    "budget_information": {{
        "budget_range": "Stated budget range if available",
        "payment_terms": "Payment structure if mentioned",
        "cost_factors": ["Factors that might affect cost"]
    }},
    "timeline_constraints": {{
        "project_duration": "Expected project duration",
        "key_milestones": ["Important deadlines or milestones"],
        "start_date": "Preferred start date if mentioned",
        "delivery_date": "Required delivery date if specified"
    }},
    "deliverables": [
        "List of expected deliverables"
    ],
    "evaluation_criteria": [
        "Criteria for proposal evaluation (highlight areas where we excel)"
    ],
    "vendor_requirements": [
        "Requirements for vendors/suppliers (note how we meet them)"
    ],
    "contact_information": {{
        "primary_contact": "Main contact person",
        "organization": "Requesting organization",
        "submission_deadline": "Proposal submission deadline"
    }},
    "risk_factors": [
        "Potential project risks identified"
    ],
    "success_metrics": [
        "How success will be measured"
    ],
    "identified_components": [
        "Key components that need specialized attention from our agents"
    ]
}}

Consider {COMPANY_PROFILE['name']}'s strengths and how we can position ourselves competitively. Ensure the response is valid JSON.
"""

def parse_rfp_with_azure_openai(rfp_text: str, config: AzureOpenAIConfig) -> Optional[Dict]:
    """Parse RFP using Azure OpenAI"""
    client = config.get_client()
    if not client:
        return None
    
    try:
        prompt = create_rfp_analysis_prompt(rfp_text)
        
        response = client.chat.completions.create(
            model=config.deployment_name,
            messages=[
                {
                    "role": "system", 
                    "content": f"You are an expert RFP analyst working for {COMPANY_PROFILE['name']}. Always respond with valid JSON format and consider our company's competitive advantages."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.1,  # Low temperature for consistent analysis
            max_tokens=4000
        )
        
        # Extract and parse JSON response
        content = response.choices[0].message.content
        
        # Print the raw response
        print("\n=== Raw Response from Azure OpenAI ===")
        print(content)
        print("====================================\n")
        
        # Clean the response to extract JSON
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            parsed_data = json.loads(json_str)
            
            # Print the parsed JSON
            print("\n=== Parsed JSON Structure ===")
            print(json.dumps(parsed_data, indent=2))
            print("============================\n")
            
            return parsed_data
        else:
            st.error("Could not extract valid JSON from Azure OpenAI response")
            return None
            
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Azure OpenAI API error: {str(e)}")
        return None

def validate_parsed_data(parsed_data: Dict) -> Dict:
    """Validate and clean parsed data"""
    required_fields = [
        'project_overview', 'technical_requirements', 'functional_requirements',
        'compliance_requirements', 'budget_information', 'timeline_constraints',
        'deliverables', 'evaluation_criteria', 'vendor_requirements',
        'contact_information', 'risk_factors', 'success_metrics', 'identified_components'
    ]
    
    # Ensure all required fields exist
    for field in required_fields:
        if field not in parsed_data:
            if field in ['project_overview', 'budget_information', 'timeline_constraints', 'contact_information']:
                parsed_data[field] = {}
            else:
                parsed_data[field] = []
    
    # Ensure identified_components has at least some default components
    if not parsed_data['identified_components']:
        parsed_data['identified_components'] = [
            "Technical Architecture", "Project Management", "Quality Assurance",
            "Compliance & Security", "Cost Estimation", "Timeline Planning"
        ]
    
    return parsed_data

# Updated Step 2: Azure OpenAI Parsing
def render_azure_openai_parsing_step():
    """Render the Azure OpenAI parsing step - UPDATED with company context"""
    st.title("🔍 Parsing RFP Document with Azure OpenAI")
    
    # Add company banner
    st.markdown(f"""
    <div class="company-info">
        <h3>🏢 Analysis by {COMPANY_PROFILE['name']}</h3>
        <p>Leveraging our expertise in {COMPANY_PROFILE['industry']} to analyze your RFP</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**Processing:** {st.session_state.rfp_name}")
    
    # Initialize Azure OpenAI config
    config = AzureOpenAIConfig()
    
    # Check if we have the file content or need to extract it
    if not hasattr(st.session_state, 'rfp_text'):
        if hasattr(st.session_state, 'uploaded_file'):
            st.info("📄 Extracting text from uploaded file...")
            extracted_text = extract_text_from_file(st.session_state.uploaded_file)
            if extracted_text:
                st.session_state.rfp_text = extracted_text
                st.success("✅ Text extraction completed!")
            else:
                st.error("❌ Failed to extract text from file")
                return
        elif hasattr(st.session_state, 'rfp_content'):
            # Tutorial mode - use sample content
            st.session_state.rfp_text = st.session_state.rfp_content
        else:
            st.error("❌ No file content available for parsing")
            return
    
    # Show text preview
    with st.expander("📄 Extracted Text Preview"):
        preview_text = st.session_state.rfp_text[:1000] + "..." if len(st.session_state.rfp_text) > 1000 else st.session_state.rfp_text
        st.text_area("Document Content", preview_text, height=200, disabled=True)
    
    parsing_container = st.container()
    
    with parsing_container:
        # Check Azure OpenAI configuration
        if not config.api_key or not config.endpoint:
            st.warning(f"⚠️ Azure OpenAI not configured. Using mock parsing for {COMPANY_PROFILE['name']} demonstration.")
            
            # Fallback to mock parsing
            st.markdown(f"""
            <div class="parsing-animation" style="padding: 2rem; border-radius: 10px; text-align: center; color: white; margin: 1rem 0;">
                <h3>🤖 Mock AI Analysis by {COMPANY_PROFILE['name']} (Azure OpenAI not configured)...</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Simulate progress for demo (ensure 0.0-1.0 range)
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            for i in range(101):
                progress_value = i / 100.0  # Convert to 0.0-1.0 range
                progress_bar.progress(progress_value)
                status_text.text(f"Mock parsing progress: {i}%")
                if i % 20 == 0:
                    time.sleep(0.1)
            
            # Use mock data for tutorial
            if st.session_state.tutorial_mode and hasattr(st.session_state, 'identified_components'):
                parsed_data = {
                    "project_overview": {
                        "title": "Sample Project",
                        "description": f"Mock analysis results by {COMPANY_PROFILE['name']}",
                        "type": "Software Development"
                    },
                    "identified_components": st.session_state.identified_components
                }
            else:
                parsed_data = {
                    "project_overview": {
                        "title": "Parsed Project",
                        "description": f"Analysis completed by {COMPANY_PROFILE['name']} with mock data",
                        "type": "Software Development"
                    },
                    "identified_components": [
                        "Technical Requirements", "Project Scope", "Timeline Constraints",
                        "Budget Parameters", "Compliance Requirements", "Deliverables"
                    ]
                }
        else:
            # Real Azure OpenAI parsing
            st.markdown(f"""
            <div class="parsing-animation" style="padding: 2rem; border-radius: 10px; text-align: center; color: white; margin: 1rem 0;">
                <h3>🧠 AI analyzing your document...</h3>
            </div>
            """, unsafe_allow_html=True)
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            details_text = st.empty()
            
            # Parsing steps with Azure OpenAI
            parsing_steps = [
                f"Connecting to {COMPANY_PROFILE['name']}'s Azure OpenAI...",
                "Extracting project overview...",
                "Identifying technical requirements against our expertise...",
                "Analyzing compliance needs with our certifications...",
                "Extracting timeline information...",
                "Analyzing budget parameters...",
                "Identifying deliverables...",
                "Generating component breakdown for our specialist agents..."
            ]
            
            # Show initial progress (ensure 0.0-1.0 range)
            for i in range(30):
                progress_value = i / 100.0  # Convert to 0.0-1.0 range
                progress_bar.progress(progress_value)
                status_text.text(f"Preparing analysis: {i}%")
                if i % 10 == 0 and i > 0:
                    step_index = min((i // 10) - 1, len(parsing_steps) - 1)
                    details_text.text(f"Step: {parsing_steps[step_index]}")
                time.sleep(0.05)
            
            # Actual Azure OpenAI call
            details_text.text(f"Calling {COMPANY_PROFILE['name']}'s Azure OpenAI API...")
            parsed_data = parse_rfp_with_azure_openai(st.session_state.rfp_text, config)
            
            if parsed_data:
                # Validate and clean the data
                parsed_data = validate_parsed_data(parsed_data)
                
                # Complete the progress animation (ensure 0.0-1.0 range)
                for i in range(30, 101):
                    progress_value = i / 100.0  # Convert to 0.0-1.0 range
                    progress_bar.progress(progress_value)
                    status_text.text(f"Processing results: {i}%")
                    if i % 15 == 0:
                        step_index = min(4 + (i // 15), len(parsing_steps) - 1)
                        details_text.text(f"Step: {parsing_steps[step_index]}")
                    time.sleep(0.03)
            else:
                st.error(f"❌ Failed to parse RFP with {COMPANY_PROFILE['name']}'s Azure OpenAI")
                return
        
        st.success(f"✅ RFP Analysis complete by {COMPANY_PROFILE['name']}! Document parsed and components identified.")
        
        # Store parsed data in session state
        st.session_state.parsed_rfp_data = parsed_data
        
        # Display analysis results
        st.markdown("### 📊 Analysis Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Project Overview")
            if parsed_data.get('project_overview'):
                overview = parsed_data['project_overview']
                st.markdown(f"**Title:** {overview.get('title', 'Not specified')}")
                st.markdown(f"**Type:** {overview.get('type', 'Not specified')}")
                if overview.get('description'):
                    st.markdown(f"**Description:** {overview['description'][:200]}...")
            
            st.markdown(f"#### 🎯 Key Components for {COMPANY_PROFILE['name']}'s Agents")
            components = parsed_data.get('identified_components', [])
            for i, component in enumerate(components[:6]):  # Show first 6
                st.markdown(f"✅ {component}")
            if len(components) > 6:
                st.markdown(f"... and {len(components) - 6} more")
        
        with col2:
            # Show statistics
            st.markdown("#### 📈 Analysis Statistics")
            
            tech_reqs = len(parsed_data.get('technical_requirements', []))
            func_reqs = len(parsed_data.get('functional_requirements', []))
            compliance_reqs = len(parsed_data.get('compliance_requirements', []))
            deliverables = len(parsed_data.get('deliverables', []))
            
            st.metric("Technical Requirements", tech_reqs)
            st.metric("Functional Requirements", func_reqs)
            st.metric("Compliance Items", compliance_reqs)
            st.metric("Deliverables", deliverables)
        
        # Show detailed breakdown in expandable sections
        with st.expander(f"🔍 Detailed Analysis Results by {COMPANY_PROFILE['name']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Technical Requirements:**")
                for req in parsed_data.get('technical_requirements', [])[:5]:
                    st.markdown(f"• {req}")
                
                st.markdown("**Compliance Requirements:**")
                for req in parsed_data.get('compliance_requirements', [])[:5]:
                    st.markdown(f"• {req}")
            
            with col2:
                st.markdown("**Timeline Information:**")
                timeline = parsed_data.get('timeline_constraints', {})
                if timeline.get('project_duration'):
                    st.markdown(f"• Duration: {timeline['project_duration']}")
                if timeline.get('delivery_date'):
                    st.markdown(f"• Delivery: {timeline['delivery_date']}")
                
                st.markdown("**Budget Information:**")
                budget = parsed_data.get('budget_information', {})
                if budget.get('budget_range'):
                    st.markdown(f"• Range: {budget['budget_range']}")
        
        if st.button(f"🤖 Dispatch to {COMPANY_PROFILE['name']} Agents", type="primary"):
            st.session_state.step = 'agent_grid'
            # Initialize first agent
            st.session_state.agents_workflow["Proposal Orchestrator Agent"]["status"] = "active"
            st.session_state.agents_workflow["Proposal Orchestrator Agent"]["progress"] = 0
            st.rerun()

# Add this to your Step 1 (Upload) section for configuration help
def add_config_help_to_upload():
    """Simplified config status"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Azure OpenAI")
    
    config = AzureOpenAIConfig()
    
    if config.api_key and config.endpoint:
        st.sidebar.success("✅ Configured")
    else:
        st.sidebar.warning("⚠️ Not Set")
        st.sidebar.caption("Set environment variables:")
        st.sidebar.code("AZURE_OPENAI_API_KEY")
        st.sidebar.code("AZURE_OPENAI_ENDPOINT")

# Sample RFP documents for tutorial - UPDATED with company context
SAMPLE_DOCUMENTS = {
    "E-commerce Platform Development": {
        "filename": "ecommerce_rfp_sample.pdf",
        "content": """
        REQUEST FOR PROPOSAL - E-COMMERCE PLATFORM DEVELOPMENT
        
        Project Overview:
        We are seeking a development partner to create a modern, scalable e-commerce platform 
        for our retail business. The platform should support B2C transactions, inventory management,
        and customer relationship management.
        
        Technical Requirements:
        - Web-based application with mobile responsiveness
        - Payment gateway integration (Stripe, PayPal)
        - Inventory management system
        - Customer portal with order tracking
        - Admin dashboard for business operations
        - SEO optimization capabilities
        
        Compliance Requirements:
        - PCI DSS compliance for payment processing
        - GDPR compliance for EU customers
        - SOC 2 Type II certification preferred
        
        Budget: $75,000 - $125,000
        Timeline: 8-10 months
        Expected Go-Live: Q3 2025
        """,
        "components": ["Web Development", "Payment Integration", "Security Compliance", "Mobile App"]
    },
    "Healthcare Management System": {
        "filename": "healthcare_rfp_sample.pdf", 
        "content": """
        REQUEST FOR PROPOSAL - HEALTHCARE MANAGEMENT SYSTEM
        
        Project Scope:
        Development of a comprehensive healthcare management system for a mid-size clinic
        network. The system should streamline patient management, appointment scheduling,
        and medical records management.
        
        Core Features Required:
        - Patient registration and management
        - Appointment scheduling system
        - Electronic Health Records (EHR)
        - Billing and insurance processing
        - Reporting and analytics dashboard
        - Telemedicine capabilities
        
        Compliance & Security:
        - HIPAA compliance mandatory
        - End-to-end encryption for patient data
        - Audit trails for all data access
        - Role-based access control
        
        Integration Requirements:
        - Integration with existing lab systems
        - Insurance verification APIs
        - Pharmacy management systems
        
        Budget: $150,000 - $200,000
        Timeline: 12-15 months
        """,
        "components": ["EHR System", "HIPAA Compliance", "Integration APIs", "Telemedicine"]
    },
    "Educational Learning Platform": {
        "filename": "education_rfp_sample.pdf",
        "content": """
        REQUEST FOR PROPOSAL - ONLINE LEARNING PLATFORM
        
        Project Description:
        We need a comprehensive online learning platform to support our educational institution's
        digital transformation. The platform should support various learning modalities and
        provide robust analytics for educators.
        
        Key Features:
        - Course management system
        - Video streaming and content delivery
        - Interactive assessments and quizzes
        - Student progress tracking
        - Discussion forums and collaboration tools
        - Mobile learning application
        
        Technical Specifications:
        - Cloud-based architecture (AWS/Azure)
        - Support for 10,000+ concurrent users
        - Multi-language support
        - Accessibility compliance (WCAG 2.1)
        - Integration with existing student information system
        
        Budget Range: $100,000 - $150,000
        Project Duration: 10-12 months
        Launch Target: Fall 2025 semester
        """,
        "components": ["LMS Development", "Video Platform", "Mobile App", "Analytics Dashboard"]
    }
}

# Initialize session state - UPDATED to remove final orchestrator
if 'step' not in st.session_state:
    st.session_state.step = 'upload'
if 'tutorial_mode' not in st.session_state:
    st.session_state.tutorial_mode = False
if 'current_agent' not in st.session_state:
    st.session_state.current_agent = None
if 'agents_workflow' not in st.session_state:
    st.session_state.agents_workflow = {
        "Proposal Orchestrator Agent": {
            "task": f"Orchestrating proposal generation for {COMPANY_PROFILE['name']}",
            "status": "pending",
            "progress": 0,
            "output": "",
            "details": f"Analyzing RFP structure and identifying key components for {COMPANY_PROFILE['name']}'s specialized agents",
            "estimated_time": "2-3 minutes",
            "feedback_requested": False,
            "human_feedback": "",
            "feedback_incorporated": False
        },
        "Tech Lead Agent": {
            "task": f"Technical architecture leveraging {COMPANY_PROFILE['name']}'s expertise",
            "status": "pending",
            "progress": 0,
            "output": "",
            "details": f"Creating technical architecture using {COMPANY_PROFILE['name']}'s proven technology stack and AWS partnership",
            "estimated_time": "8-10 minutes",
            "feedback_requested": False,
            "human_feedback": "",
            "feedback_incorporated": False
        },
        "Estimation Agent": {
            "task": f"Cost estimation based on {COMPANY_PROFILE['name']}'s 200+ projects",
            "status": "pending",
            "progress": 0,
            "output": "",
            "details": f"Calculating project costs using {COMPANY_PROFILE['name']}'s historical data and 15% faster delivery advantage",
            "estimated_time": "5-7 minutes",
            "feedback_requested": False,
            "human_feedback": "",
            "feedback_incorporated": False
        },
        "Timeline Agent": {
            "task": f"Project timeline using {COMPANY_PROFILE['name']}'s Agile methodology",
            "status": "pending",
            "progress": 0,
            "output": "",
            "details": f"Developing project schedules with {COMPANY_PROFILE['name']}'s proven weekly demos and 24/7 support",
            "estimated_time": "4-6 minutes",
            "feedback_requested": False,
            "human_feedback": "",
            "feedback_incorporated": False
        },
        "Legal & Compliance Agent": {
            "task": f"Legal review leveraging {COMPANY_PROFILE['name']}'s certifications",
            "status": "pending",
            "progress": 0,
            "output": "",
            "details": f"Ensuring compliance using {COMPANY_PROFILE['name']}'s ISO 27001 and SOC 2 certifications",
            "estimated_time": "6-8 minutes",
            "feedback_requested": False,
            "human_feedback": "",
            "feedback_incorporated": False
        },
        "Sales/Marketing Agent": {
            "task": f"Value propositions highlighting {COMPANY_PROFILE['name']}'s advantages",
            "status": "pending",
            "progress": 0,
            "output": "",
            "details": f"Crafting compelling proposals showcasing {COMPANY_PROFILE['name']}'s 99.5% retention rate and awards",
            "estimated_time": "4-5 minutes",
            "feedback_requested": False,
            "human_feedback": "",
            "feedback_incorporated": False
        }
        # REMOVED: "Proposal Orchestrator Agent (Final)"
    }
if 'consolidated_document' not in st.session_state:
    st.session_state.consolidated_document = ""
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []

# Sidebar navigation
st.sidebar.title(f"🚀 {COMPANY_PROFILE['name']}")
st.sidebar.caption(f"{COMPANY_PROFILE['industry']}")

# Tutorial mode toggle
if st.sidebar.checkbox("🎓 Tutorial Mode", value=st.session_state.tutorial_mode):
    st.session_state.tutorial_mode = True
    if st.session_state.tutorial_mode:
        st.sidebar.markdown(f"""
        <div class="tutorial-banner">
            <h4>🎓 Tutorial Mode Active</h4>
            <p>Experience {COMPANY_PROFILE['name']}'s workflow with sample data</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.session_state.tutorial_mode = False

st.sidebar.markdown("---")

steps = ["📄 Upload RFP", "🔍 Parsing", "🤖 Agent Grid", "💬 Feedback", "✅ Check Work", "📋 Consolidate"]
current_step_index = ["upload", "parsing", "agent_grid", "feedback", "agent_check", "consolidate"].index(st.session_state.step)

for i, step in enumerate(steps):
    if i == current_step_index:
        st.sidebar.markdown(f"**➤ {step}**")
    elif i < current_step_index:
        st.sidebar.markdown(f"✅ {step}")
    else:
        st.sidebar.markdown(f"⚪ {step}")

st.sidebar.markdown("---")

# Step 1: Upload RFP document - UPDATED with company context
if st.session_state.step == 'upload':
    st.title("📄 Upload RFP Document")
    
    # Add company banner
    st.markdown(f"""
    <div class="company-info">
        <h3>🏢 Welcome to {COMPANY_PROFILE['name']}</h3>
        <p><strong>Specializing in:</strong> {', '.join(COMPANY_PROFILE['specializations'][:3])}</p>
        <p><strong>Track Record:</strong> {COMPANY_PROFILE['portfolio_highlights'][0]} with {COMPANY_PROFILE['portfolio_highlights'][2]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.tutorial_mode:
        st.markdown(f"""
        <div class="tutorial-banner">
            <h3>🎓 Tutorial Mode: {COMPANY_PROFILE['name']} Sample Documents</h3>
            <p>Choose from our sample RFP documents to experience {COMPANY_PROFILE['name']}'s complete workflow</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"Welcome to {COMPANY_PROFILE['name']}'s AI-powered Proposal Generator! Start by uploading your RFP document or try our tutorial.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.tutorial_mode:
            st.markdown("### 🎓 Choose a Sample Document")
            selected_sample = st.selectbox(
                "Select a sample RFP to explore:",
                list(SAMPLE_DOCUMENTS.keys()),
                help=f"These are realistic RFP examples to demonstrate {COMPANY_PROFILE['name']}'s system"
            )
            
            if selected_sample:
                sample_doc = SAMPLE_DOCUMENTS[selected_sample]
                st.markdown(f"**Sample Document:** {sample_doc['filename']}")
                
                with st.expander("📄 Preview Sample Content"):
                    st.text_area("RFP Content Preview", sample_doc['content'], height=200, disabled=True)
                
                if st.button(f"🚀 Start Tutorial with {COMPANY_PROFILE['name']}", type="primary"):
                    st.session_state.rfp_name = sample_doc['filename']
                    st.session_state.rfp_content = sample_doc['content']
                    st.session_state.identified_components = sample_doc['components']
                    st.session_state.step = 'parsing'
                    st.rerun()
        else:
            uploaded_file = st.file_uploader(
                f"Choose an RFP document for {COMPANY_PROFILE['name']} to analyze", 
                type=["pdf", "docx", "txt"],
                help="Supported formats: PDF, DOCX, TXT"
            )
            
            if uploaded_file is not None:
                st.success(f"✅ File uploaded for {COMPANY_PROFILE['name']} analysis: {uploaded_file.name}")
                st.session_state.rfp_name = uploaded_file.name
                st.session_state.uploaded_file = uploaded_file 
                
                file_details = {
                    "Filename": uploaded_file.name,
                    "File size": f"{uploaded_file.size} bytes",
                    "File type": uploaded_file.type
                }
                
                st.markdown("**File Details:**")
                for key, value in file_details.items():
                    st.markdown(f"- **{key}:** {value}")
                
                if st.button(f"🚀 Start {COMPANY_PROFILE['name']} Processing", type="primary"):
                    st.session_state.step = 'parsing'
                    st.rerun()
    
    with col2:
        st.markdown(f"### 📋 {COMPANY_PROFILE['name']}'s Process")
        st.markdown(f"""
        1. **Document Parsing** - Our AI analyzes your RFP
        2. **Agent Dispatch** - {COMPANY_PROFILE['name']}'s specialized agents work
        3. **Human Feedback** - Review and guide agent work
        4. **Content Generation** - Each agent creates their section
        5. **Quality Check** - Review and refine outputs  
        6. **Final Proposal** - Export your complete proposal
        """)
        
        # Show company highlights
        st.markdown(f"### 🏆 Why Choose {COMPANY_PROFILE['name']}?")
        st.markdown(f"""
        - **{COMPANY_PROFILE['portfolio_highlights'][0]}**
        - **{COMPANY_PROFILE['portfolio_highlights'][2]}**
        - **{COMPANY_PROFILE['recent_achievements'][0][:50]}...**
        """)
        
        if not st.session_state.tutorial_mode:
            st.markdown("---")
            st.markdown("### 🎓 New to our system?")
            if st.button("Try Tutorial Mode"):
                st.session_state.tutorial_mode = True
                st.rerun()
    add_config_help_to_upload()

# Step 2: Parsing simulation
elif st.session_state.step == 'parsing':
    render_azure_openai_parsing_step()

# Step 3: Agent Grid
elif st.session_state.step == 'agent_grid':
    render_manual_langgraph_ui()

# Step 4: Human Feedback - UPDATED to remove final orchestrator references
elif st.session_state.step == 'feedback':
    st.title("💬 Provide Human Feedback")
    st.markdown(f"Guide {COMPANY_PROFILE['name']}'s AI agents with your expertise and domain knowledge")
    
    # Find agents requesting feedback
    feedback_agents = {name: info for name, info in st.session_state.agents_workflow.items() 
                      if info["feedback_requested"] and not info["feedback_incorporated"]}
    
    if not feedback_agents:
        st.info(f"No {COMPANY_PROFILE['name']} agents are currently requesting feedback.")
        if st.button("🔙 Back to Agent Grid"):
            st.session_state.step = 'agent_grid'
            st.rerun()
    else:
        selected_feedback_agent = st.selectbox(
            f"Select {COMPANY_PROFILE['name']} agent to provide feedback:",
            list(feedback_agents.keys()),
            format_func=lambda x: f"🔔 {x} - Requesting Feedback"
        )
        
        agent_info = feedback_agents[selected_feedback_agent]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### 🤖 {selected_feedback_agent}")
            st.markdown(f"**Current Task:** {agent_info['task']}")
            st.markdown(f"**Progress:** {agent_info['progress']}%")
            st.markdown(f"**Details:** {agent_info['details']}")
            
            # Show current work/direction - UPDATED with company context
            st.markdown("### 🔍 Current Work Direction")
            current_work = {
                "Proposal Orchestrator Agent": f"Identified 6 main RFP components using {COMPANY_PROFILE['name']}'s proven analysis framework. Planning to dispatch technical requirements to Tech Lead Agent first, followed by parallel processing of estimation and timeline.",
                "Tech Lead Agent": f"Recommending microservices architecture with React frontend and Node.js backend leveraging {COMPANY_PROFILE['name']}'s AWS Advanced Consulting Partner status. Considering containerized deployment with our proven security framework.",
                "Estimation Agent": f"Initial cost estimate at $85,000 based on {COMPANY_PROFILE['name']}'s 200+ project database. Factoring in our 15% faster delivery advantage and 15% risk buffer.",
                "Timeline Agent": f"Proposing 26-week timeline using {COMPANY_PROFILE['name']}'s Agile methodology with 4-week planning phase, 16-week development, and 6-week testing/deployment with weekly client demos.",
                "Legal & Compliance Agent": f"Reviewing GDPR and SOC2 requirements leveraging {COMPANY_PROFILE['name']}'s ISO 27001 and SOC 2 Type II certifications. Identifying standard contract terms based on our Fortune 500 client experience.",
                "Sales/Marketing Agent": f"Focusing on {COMPANY_PROFILE['name']}'s scalability advantages and modern technology stack. Emphasizing our 99.5% client retention rate and 2024 'Best Custom Software Developer' award."
            }
            
            st.info(current_work.get(selected_feedback_agent, f"Working on assigned tasks for {COMPANY_PROFILE['name']}..."))
            
            # Feedback form
            st.markdown("### 💭 Your Feedback")
            feedback_type = st.radio(
                "Type of feedback:",
                ["Direction Guidance", "Technical Correction", "Business Insight", "Risk Concern", "Additional Requirements"]
            )
            
            feedback_text = st.text_area(
                "Provide your feedback:",
                placeholder=f"Share your insights for {COMPANY_PROFILE['name']}'s proposal, corrections, or additional requirements...",
                height=150
            )
            
            # Feedback priority
            priority = st.select_slider(
                "Feedback Priority:",
                options=["Low", "Medium", "High", "Critical"],
                value="Medium"
            )
            
            if st.button("📤 Submit Feedback", type="primary"):
                if feedback_text.strip():
                    # Store feedback
                    feedback_entry = {
                        "agent": selected_feedback_agent,
                        "type": feedback_type,
                        "content": feedback_text,
                        "priority": priority,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    st.session_state.feedback_history.append(feedback_entry)
                    agent_info["human_feedback"] = feedback_text
                    agent_info["feedback_incorporated"] = True
                    agent_info["status"] = "active"  # Resume work with feedback
                    
                    st.success(f"✅ Feedback submitted to {COMPANY_PROFILE['name']}'s {selected_feedback_agent}!")
                    time.sleep(1)
                    st.session_state.step = 'agent_grid'
                    st.rerun()
                else:
                    st.error("Please provide feedback before submitting.")
        
        with col2:
            st.markdown("### 📊 Feedback Guidelines")
            st.markdown(f"""
            **Direction Guidance:**
            - Suggest alternative approaches for {COMPANY_PROFILE['name']}
            - Recommend specific technologies from our stack
            - Guide architectural decisions
            
            **Technical Correction:**
            - Point out technical inaccuracies
            - Suggest better solutions using our expertise
            - Correct assumptions
            
            **Business Insight:**
            - Share domain knowledge
            - Highlight business priorities
            - Suggest value propositions for {COMPANY_PROFILE['name']}
            
            **Risk Concern:**
            - Identify potential risks
            - Suggest mitigation strategies
            - Flag compliance issues
            """)
            
            # Show feedback history
            if st.session_state.feedback_history:
                st.markdown("### 📝 Recent Feedback")
                for feedback in st.session_state.feedback_history[-3:]:
                    st.markdown(f"""
                    <div class="feedback-section">
                        <strong>{feedback['agent']}</strong><br>
                        <em>{feedback['type']} - {feedback['priority']}</em><br>
                        {feedback['content'][:100]}...
                    </div>
                    """, unsafe_allow_html=True)

# Step 5: Check Agent Work - UPDATED to remove final orchestrator and add company context
elif st.session_state.step == 'agent_check':
    st.title("✅ Review Agent Work")
    st.markdown(f"Examine the output from each of {COMPANY_PROFILE['name']}'s specialized agents and add to your proposal")
    
    # Agent selector - filter out final orchestrator if it exists
    available_agents = [name for name in st.session_state.agents_workflow.keys() if "Final" not in name]
    
    selected_agent = st.selectbox(
        f"Select a {COMPANY_PROFILE['name']} agent to review their work:",
        available_agents,
        format_func=lambda x: f"{'✅' if st.session_state.agents_workflow[x]['status'] == 'completed' else '🔄'} {x}"
    )
    
    agent_info = st.session_state.agents_workflow[selected_agent]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### 🤖 {selected_agent}")
        st.markdown(f"**Task:** {agent_info['task']}")
        st.markdown(f"**Details:** {agent_info['details']}")
        
        # Show human feedback if provided
        if agent_info['human_feedback']:
            st.markdown("### 💭 Human Feedback Incorporated")
            st.info(f"Feedback: {agent_info['human_feedback']}")
        
        # Generate mock output if not exists - UPDATED with company context
        if agent_info['output'] == "":
            mock_outputs = {
                "Proposal Orchestrator Agent": f"""
**RFP Analysis Summary by {COMPANY_PROFILE['name']}**

**Project Understanding:**
- Project Type: Custom Software Development aligned with {COMPANY_PROFILE['name']}'s core specializations
- Key Requirements: Web application with mobile responsiveness - perfect match for our expertise
- Compliance Needs: GDPR, SOC2 Type II - leveraging our existing ISO 27001 and SOC 2 certifications
- Budget Range: $50,000 - $100,000 - within our sweet spot for enterprise solutions
- Timeline: 6-8 months - achievable with our 15% faster delivery track record

**{COMPANY_PROFILE['name']}'s Competitive Positioning:**
- Direct alignment with our Enterprise Software Development specialization
- Opportunity to showcase our AWS Advanced Consulting Partner status
- Leverage our 200+ successful projects experience
- Highlight our 99.5% client retention rate
                """,
                "Tech Lead Agent": f"""
**Technical Architecture Proposal by {COMPANY_PROFILE['name']}**

**Recommended Technology Stack (Leveraging Our Expertise):**
- Frontend: React.js with TypeScript (from our proven stack)
- Backend: Node.js with Express.js (aligned with our specializations)
- Database: PostgreSQL with Redis caching (enterprise-grade solutions)
- Cloud: AWS with containerized deployment (leveraging our AWS Advanced Partner status)
- Security: OAuth 2.0, JWT tokens, SSL/TLS (aligned with our ISO 27001 certification)

**{COMPANY_PROFILE['name']}'s System Architecture:**
- Microservices architecture for scalability (proven in our 200+ projects)
- RESTful API design with GraphQL for complex queries
- Automated CI/CD pipeline with testing (part of our DevOps specialization)
- Monitoring and logging with CloudWatch (AWS partnership benefits)
- 99.9% uptime guarantee backed by our track record

**Security Implementation:**
- Leveraging our SOC 2 Type II compliance expertise
- End-to-end encryption protocols
- Multi-factor authentication integration
                """,
                "Estimation Agent": f"""
**Project Estimation by {COMPANY_PROFILE['name']}**

**Development Effort (Based on Our 200+ Project Database):**
- Frontend Development: 320 hours ($32,000) - optimized with our React expertise
- Backend Development: 280 hours ($28,000) - efficient with our Node.js specialization
- Database Design: 80 hours ($8,000) - streamlined with our enterprise experience
- Testing & QA: 120 hours ($12,000) - comprehensive with our proven methodologies
- DevOps & Deployment: 60 hours ($6,000) - accelerated with our AWS partnership

**{COMPANY_PROFILE['name']} Value Proposition:**
- **Total Estimated Cost: $86,000** (competitive pricing)
- **Timeline: 22 weeks** (15% faster than industry standard)
- **Risk Buffer: 10%** (reduced due to our proven track record)
- **Post-launch Support: 6 months included** (part of our 24/7 global coverage)

**Cost Advantages:**
- 15% faster delivery = reduced total cost
- Proprietary frameworks reduce development time
- AWS partnership provides infrastructure cost savings
                """,
                "Timeline Agent": f"""
**Project Timeline by {COMPANY_PROFILE['name']}**

**{COMPANY_PROFILE['name']}'s Proven Agile Methodology:**

**Phase 1: Planning & Design (Weeks 1-3)**
- Requirements analysis using our proven framework
- System design leveraging our architectural expertise
- UI/UX mockups with our design team
- Weekly client demos (signature {COMPANY_PROFILE['name']} approach)

**Phase 2: Development (Weeks 4-18)**
- Sprint 1-4: Core functionality development
- Sprint 5-8: Advanced features implementation  
- Sprint 9-12: Integration & API development
- Sprint 13-15: Testing & optimization
- **Weekly demos and feedback sessions** (our proven engagement model)

**Phase 3: Deployment & Launch (Weeks 19-22)**
- User acceptance testing with client team
- Performance optimization and security hardening
- Production deployment with AWS best practices
- Knowledge transfer and documentation
- Go-live support with our 24/7 coverage

**{COMPANY_PROFILE['name']}'s Timeline Advantages:**
- 15% faster delivery than industry standard
- Weekly client engagement reduces late-stage changes
- Proven risk mitigation strategies from 200+ projects
                """,
                "Legal & Compliance Agent": f"""
**Legal & Compliance Assessment by {COMPANY_PROFILE['name']}**

**Contract Terms (Based on Our Fortune 500 Experience):**
- Intellectual property rights clearly defined per our standard agreements
- Data protection clauses leveraging our GDPR compliance expertise  
- Liability limitations based on our insurance and legal framework
- Payment terms: 30% upfront, 40% at milestone, 30% completion (our proven structure)

**Compliance Requirements (Leveraging Our Certifications):**
- **GDPR compliance** - {COMPANY_PROFILE['name']} is fully GDPR compliant
- **SOC2 Type II certification** - we maintain active certification
- **ISO 27001 compliance** - part of our core security framework
- Regular security audits (quarterly internal, annual external)
- Data encryption at rest and in transit (standard in all our projects)

**{COMPANY_PROFILE['name']}'s Legal Advantages:**
- **Risk Assessment: LOW** (due to our proven compliance track record)
- Existing insurance coverage for all project types
- Legal framework tested across 200+ successful projects
- Established relationships with compliance auditors
- Template agreements refined through Fortune 500 engagements

**Regulatory Expertise:**
- Financial services compliance (for payment processing)
- Healthcare data protection (HIPAA experience)
- International data transfer protocols
                """,
                "Sales/Marketing Agent": f"""
**Executive Summary & Value Proposition by {COMPANY_PROFILE['name']}**

**Why Choose {COMPANY_PROFILE['name']} for Your Project:**

**Proven Track Record:**
- **{COMPANY_PROFILE['portfolio_highlights'][0]}** across diverse industries
- **{COMPANY_PROFILE['portfolio_highlights'][2]}** - demonstrating client satisfaction
- **{COMPANY_PROFILE['recent_achievements'][0]}** - industry recognition

**Technical Excellence:**
- **{COMPANY_PROFILE['specializations'][0]}** - perfect match for your requirements
- **AWS Advanced Consulting Partner** - ensuring optimal cloud solutions
- **ISO 27001 & SOC 2 Type II certified** - guaranteeing security and compliance
- Modern technology stack with proven scalability

**Competitive Advantages:**
- **15% faster development** using our proprietary frameworks and methodologies
- **99.9% uptime guarantee** with our hosting and support solution
- **24/7 global support coverage** across our international offices
- **Weekly client demos** ensuring transparency and alignment

**Business Benefits:**
- Scalable architecture supporting future growth (tested in 200+ projects)
- Modern, responsive design for optimal user experience
- Robust security measures protecting sensitive data
- Comprehensive testing ensuring reliability and performance

**{COMPANY_PROFILE['name']}'s Unique Value:**
- **40% operational cost savings** for clients (proven across our portfolio)
- **99.5% client retention rate** - clients stay with us long-term
- **End-to-end solution delivery** - from concept to post-launch support
- **AI-assisted development frameworks** - cutting-edge efficiency

**Investment Protection:**
- Fixed-price proposal with no hidden costs
- 6-month post-launch support included
- Future enhancement roadmap provided
- Knowledge transfer ensuring client independence

**Next Steps with {COMPANY_PROFILE['name']}:**
1. Proposal review and technical deep-dive session
2. Contract negotiation with our legal team
3. Project kickoff with dedicated team introduction
4. Development commencement with immediate weekly demos

*Partner with {COMPANY_PROFILE['name']} - where innovation meets reliability.*
                """
            }
            
            # Incorporate feedback into output if available
            base_output = mock_outputs.get(selected_agent, f"No output available from {COMPANY_PROFILE['name']} agent.")
            if agent_info['human_feedback']:
                base_output += f"\n\n**Note:** This output has been refined by {COMPANY_PROFILE['name']} based on your feedback: {agent_info['human_feedback'][:100]}..."
            
            agent_info['output'] = base_output
        
        st.markdown(f"### 📄 {selected_agent} Output")
        st.markdown(agent_info['output'])
        
        # Feedback option for completed work
        if agent_info['status'] == 'completed':
            with st.expander("💬 Provide Additional Feedback"):
                additional_feedback = st.text_area(
                    f"Any additional feedback for this {COMPANY_PROFILE['name']} output?",
                    placeholder="Suggest improvements, corrections, or additions..."
                )
                if st.button(f"📤 Submit Additional Feedback for {selected_agent}"):
                    if additional_feedback.strip():
                        feedback_entry = {
                            "agent": selected_agent,
                            "type": "Post-completion Review",
                            "content": additional_feedback,
                            "priority": "Medium",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.session_state.feedback_history.append(feedback_entry)
                        st.success("Additional feedback recorded!")
        
        if st.button(f"✅ Add {selected_agent} output to proposal", type="primary"):
            if f"### {selected_agent}" not in st.session_state.consolidated_document:
                st.session_state.consolidated_document += f"\n\n## {selected_agent}\n{agent_info['output']}"
                st.success(f"✅ Added output from {COMPANY_PROFILE['name']}'s {selected_agent} to consolidated document.")
            else:
                st.warning(f"This {COMPANY_PROFILE['name']} agent's output is already in the consolidated document.")
    
    with col2:
        st.markdown("### 📊 Agent Status")
        st.metric("Progress", f"{agent_info['progress']}%")
        st.metric("Status", agent_info['status'].title())
        
        if agent_info['human_feedback']:
            st.markdown("### 💭 Feedback Given")
            st.success("✅ Human feedback provided")
        
        # Show which outputs are already added
        st.markdown(f"### 📋 Added to {COMPANY_PROFILE['name']} Proposal")
        for agent_name in available_agents:  # Use filtered list
            if f"### {agent_name}" in st.session_state.consolidated_document:
                st.markdown(f"✅ {agent_name}")
            else:
                st.markdown(f"⚪ {agent_name}")
    
    st.markdown("---")
    
    # Check if all outputs are added - UPDATED count
    added_count = sum(1 for agent_name in available_agents 
                     if f"### {agent_name}" in st.session_state.consolidated_document)
    
    if added_count == len(available_agents):
        st.success(f"🎉 All {COMPANY_PROFILE['name']} agent outputs have been added to the proposal!")
    
    if st.button(f"📋 Proceed to Consolidate {COMPANY_PROFILE['name']} Proposal", type="primary"):
        st.session_state.step = 'consolidate'
        st.rerun()

# Step 6: Consolidate and Export - UPDATED with company context and auto-generation
elif st.session_state.step == 'consolidate':
    st.title("📋 Consolidate and Export Proposal")
    st.markdown(f"Review your complete {COMPANY_PROFILE['name']} proposal document and export it")
    
    # Auto-generate final proposal if not already done
    if st.session_state.consolidated_document.strip() == "":
        st.info(f"🔄 Auto-generating final proposal from {COMPANY_PROFILE['name']} agents...")
        
        # Get the LangGraph system and generate final proposal
        langgraph_system = get_simple_langgraph_system()
        if hasattr(st.session_state, 'workflow_state') and st.session_state.workflow_state["agent_outputs"]:
            final_proposal = langgraph_system.generate_final_proposal(st.session_state.workflow_state)
            st.session_state.consolidated_document = final_proposal
            st.success(f"✅ Final {COMPANY_PROFILE['name']} proposal generated automatically!")
        else:
            st.warning(f"⚠️ No content from {COMPANY_PROFILE['name']} agents available. Please run agents first.")
            if st.button("⬅️ Go Back to Agent Grid"):
                st.session_state.step = 'agent_grid'
                st.rerun()
            
    
    # Document preview
    st.markdown(f"### 📄 {COMPANY_PROFILE['name']} Proposal Document Preview")
    
    full_document = st.session_state.consolidated_document
    
    # Show document in expandable text area
    st.text_area(
        f"Complete {COMPANY_PROFILE['name']} Proposal Document", 
        value=full_document, 
        height=400,
        help=f"This is your complete proposal document from {COMPANY_PROFILE['name']} ready for export"
    )
    
    # Document statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        word_count = len(full_document.split())
        st.metric("Word Count", word_count)
    with col2:
        char_count = len(full_document)
        st.metric("Characters", char_count)
    with col3:
        section_count = full_document.count('##')
        st.metric("Sections", section_count)
    with col4:
        feedback_count = len(st.session_state.feedback_history)
        st.metric("Feedback Sessions", feedback_count)
    
    st.markdown("---")
    
    # Export options
    st.markdown("### 📤 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Text export
        def convert_text_to_file(text):
            return text.encode('utf-8')
        
        proposal_bytes = convert_text_to_file(full_document)
        
        st.download_button(
            label="📄 Download as Text File",
            data=proposal_bytes,
            file_name=f"{COMPANY_PROFILE['name'].lower().replace(' ', '_')}_proposal_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            type="primary"
        )
    
    with col2:
        # Markdown export
        st.download_button(
            label="📝 Download as Markdown",
            data=proposal_bytes,
            file_name=f"{COMPANY_PROFILE['name'].lower().replace(' ', '_')}_proposal_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown"
        )
    
    # Feedback export
    if st.session_state.feedback_history:
        st.markdown("### 💭 Export Feedback History")
        feedback_df = pd.DataFrame(st.session_state.feedback_history)
        csv_data = feedback_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📊 Download Feedback History (CSV)",
            data=csv_data,
            file_name=f"{COMPANY_PROFILE['name'].lower().replace(' ', '_')}_feedback_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    # Workflow completion
    st.success(f"🎉 {COMPANY_PROFILE['name']} proposal generation completed successfully!")
    
    if st.session_state.tutorial_mode:
        st.info(f"🎓 Tutorial completed! You've experienced {COMPANY_PROFILE['name']}'s full workflow with human feedback integration.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"🔄 Start New {COMPANY_PROFILE['name']} Proposal", type="primary"):
            # Reset all session state
            for key in list(st.session_state.keys()):
                if key not in ['tutorial_mode']:  # Keep tutorial mode setting
                    del st.session_state[key]
            st.rerun()
    
    with col2:
        if st.button(f"📧 Share Feedback about {COMPANY_PROFILE['name']}"):
            st.info(f"Thank you for using {COMPANY_PROFILE['name']}'s AI Proposal Generator! Your feedback helps us improve our services.")

# Sidebar status - UPDATED with company context
st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📊 {COMPANY_PROFILE['name']} Session Info")
if 'rfp_name' in st.session_state:
    st.sidebar.markdown(f"**RFP:** {st.session_state.rfp_name}")

current_time = datetime.now().strftime("%H:%M:%S")
st.sidebar.markdown(f"**Time:** {current_time}")

if st.session_state.step in ['agent_grid', 'feedback', 'agent_check', 'consolidate']:
    completed_agents = sum(1 for agent in st.session_state.agents_workflow.values() if agent["status"] == "completed")
    total_agents = len([name for name in st.session_state.agents_workflow.keys() if "Final" not in name])  # Exclude final orchestrator
    progress_value = completed_agents / total_agents if total_agents > 0 else 0.0  # Ensure 0.0-1.0 range
    st.sidebar.progress(progress_value, text=f"Agents: {completed_agents}/{total_agents}")

# Feedback status in sidebar
if st.session_state.feedback_history:
    st.sidebar.markdown("### 💭 Feedback Status")
    st.sidebar.metric("Feedback Sessions", len(st.session_state.feedback_history))
    
    feedback_pending = sum(1 for agent in st.session_state.agents_workflow.values() if agent["feedback_requested"] and not agent["feedback_incorporated"])
    if feedback_pending > 0:
        st.sidebar.warning(f"🔔 {feedback_pending} agents need feedback")

# Company info in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"### 🏢 {COMPANY_PROFILE['name']}")
st.sidebar.markdown(f"**Industry:** {COMPANY_PROFILE['industry']}")
st.sidebar.markdown(f"**Founded:** {COMPANY_PROFILE['founded']}")
st.sidebar.markdown(f"**Team:** {COMPANY_PROFILE['employees']}")
with st.sidebar.expander("🏆 Our Achievements"):
    for achievement in COMPANY_PROFILE['recent_achievements'][:2]:
        st.sidebar.markdown(f"• {achievement}")