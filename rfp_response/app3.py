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
# State definition (same as before)
class ProposalState(TypedDict):
    rfp_data: dict
    current_agent: str
    agent_outputs: dict
    human_feedback: dict
    feedback_requests: List[str]
    completed_agents: List[str]
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_action: str

# Simplified LangGraph System without SQLite persistence
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
        """Create the LangGraph workflow - simplified without checkpointer"""
        workflow = StateGraph(ProposalState)
        
        # Add nodes for each agent
        workflow.add_node("orchestrator", self.orchestrator_agent)
        workflow.add_node("tech_lead", self.tech_lead_agent)
        workflow.add_node("estimation", self.estimation_agent)
        workflow.add_node("timeline", self.timeline_agent)
        workflow.add_node("legal", self.legal_agent)
        workflow.add_node("sales", self.sales_agent)
        workflow.add_node("final_orchestrator", self.final_orchestrator_agent)
        
        # Set entry point
        workflow.set_entry_point("orchestrator")
        
        # Simple linear workflow
        workflow.add_edge("orchestrator", "tech_lead")
        workflow.add_edge("tech_lead", "estimation")
        workflow.add_edge("estimation", "timeline")
        workflow.add_edge("timeline", "legal")
        workflow.add_edge("legal", "sales")
        workflow.add_edge("sales", "final_orchestrator")
        workflow.add_edge("final_orchestrator", END)
        
        # Compile without checkpointer - much simpler!
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
        """Sales/Marketing Agent"""
        if not self.llm:
            output = self._get_mock_output("Sales/Marketing Agent")
        else:
            prompt = self._create_sales_prompt(state["rfp_data"], state.get("human_feedback", {}))
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            output = response.content
        
        state["agent_outputs"]["Sales/Marketing Agent"] = output
        state["completed_agents"].append("Sales/Marketing Agent")
        state["current_agent"] = "Proposal Orchestrator Agent (Final)"
        
        return state
    
    def final_orchestrator_agent(self, state: ProposalState) -> ProposalState:
        """Final Orchestrator Agent"""
        if not self.llm:
            output = self._get_mock_output("Proposal Orchestrator Agent (Final)")
        else:
            all_outputs = "\n\n".join([f"## {agent}: {output}" for agent, output in state["agent_outputs"].items()])
            prompt = self._create_final_orchestrator_prompt(state["rfp_data"], all_outputs, state.get("human_feedback", {}))
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            output = response.content
        
        state["agent_outputs"]["Proposal Orchestrator Agent (Final)"] = output
        state["completed_agents"].append("Proposal Orchestrator Agent (Final)")
        state["current_agent"] = "completed"
        
        return state
    
    # All the prompt creation methods (same as before)
    def _create_orchestrator_prompt(self, rfp_data: dict) -> str:
        return f"""You are a Proposal Orchestrator Agent. Based on the RFP analysis below, create a comprehensive project breakdown.

RFP Data: {rfp_data}

Create a detailed response covering:
1. Project scope overview and understanding
2. Key components identification and prioritization  
3. Risk assessment and mitigation strategies
4. Success criteria and metrics

Format as a professional proposal section with clear headings."""
    
    def _create_tech_lead_prompt(self, rfp_data: dict, feedback: dict) -> str:
        feedback_text = feedback.get("Tech Lead Agent", "")
        feedback_context = f"\n\nHuman Feedback: {feedback_text}" if feedback_text else ""
        
        return f"""You are a Technical Lead Agent. Design the technical architecture.

RFP Data: {rfp_data}
{feedback_context}

Provide:
1. Recommended technology stack
2. System architecture design
3. Security implementation strategy
4. Development methodology

Be specific about technologies and implementation details."""
    
    def _create_estimation_prompt(self, rfp_data: dict, feedback: dict) -> str:
        feedback_text = feedback.get("Estimation Agent", "")
        feedback_context = f"\n\nHuman Feedback: {feedback_text}" if feedback_text else ""
        
        return f"""You are an Estimation Agent. Provide detailed cost estimates.

RFP Data: {rfp_data}
{feedback_context}

Provide:
1. Work breakdown structure
2. Effort estimation by component
3. Resource allocation
4. Total project cost with breakdown"""
    
    def _create_timeline_prompt(self, rfp_data: dict, feedback: dict) -> str:
        feedback_text = feedback.get("Timeline Agent", "")
        feedback_context = f"\n\nHuman Feedback: {feedback_text}" if feedback_text else ""
        
        return f"""You are a Timeline Agent. Create project timeline.

RFP Data: {rfp_data}
{feedback_context}

Develop:
1. Project phases and milestones
2. Task breakdown and dependencies
3. Resource scheduling
4. Delivery schedule with key dates"""
    
    def _create_legal_prompt(self, rfp_data: dict, feedback: dict) -> str:
        feedback_text = feedback.get("Legal & Compliance Agent", "")
        feedback_context = f"\n\nHuman Feedback: {feedback_text}" if feedback_text else ""
        
        return f"""You are a Legal & Compliance Agent. Provide legal guidance.

RFP Data: {rfp_data}
{feedback_context}

Address:
1. Regulatory compliance requirements
2. Data protection considerations
3. Contract terms recommendations
4. Risk assessment"""
    
    def _create_sales_prompt(self, rfp_data: dict, feedback: dict) -> str:
        feedback_text = feedback.get("Sales/Marketing Agent", "")
        feedback_context = f"\n\nHuman Feedback: {feedback_text}" if feedback_text else ""
        
        return f"""You are a Sales/Marketing Agent. Create value propositions.

RFP Data: {rfp_data}
{feedback_context}

Develop:
1. Executive summary
2. Company capabilities
3. Competitive advantages
4. Client benefits and ROI"""
    
    def _create_final_orchestrator_prompt(self, rfp_data: dict, all_outputs: str, feedback: dict) -> str:
        feedback_text = feedback.get("Proposal Orchestrator Agent (Final)", "")
        feedback_context = f"\n\nHuman Feedback: {feedback_text}" if feedback_text else ""
        
        return f"""You are the Final Proposal Orchestrator. Synthesize all outputs.

RFP Data: {rfp_data}
Agent Outputs: {all_outputs}
{feedback_context}

Create final synthesis that:
1. Integrates all aspects
2. Ensures consistency
3. Provides clear next steps
4. Creates compelling conclusion"""
    
    def _get_mock_output(self, agent_name: str) -> str:
        """Mock outputs when Azure OpenAI isn't configured"""
        mock_outputs = {
            "Proposal Orchestrator Agent": "**Project Analysis Complete** - Successfully analyzed RFP requirements and identified 6 key project components.",
            "Tech Lead Agent": "**Technical Architecture Designed** - Recommended modern tech stack with React frontend, Node.js backend, AWS deployment.",
            "Estimation Agent": "**Cost Analysis Complete** - Total estimate: $85,000 over 26 weeks with detailed component breakdown.",
            "Timeline Agent": "**Project Timeline Created** - 26-week schedule: Planning (4w), Development (16w), Testing (4w), Deployment (2w).",
            "Legal & Compliance Agent": "**Compliance Review Complete** - Addressed GDPR, SOC2 requirements with low risk assessment.",
            "Sales/Marketing Agent": "**Value Proposition Developed** - Created compelling executive summary highlighting competitive advantages.",
            "Proposal Orchestrator Agent (Final)": "**Final Synthesis Complete** - Integrated all sections into cohesive proposal with clear next steps."
        }
        return mock_outputs.get(agent_name, f"Mock output for {agent_name}")
    
    def run_single_agent(self, agent_name: str, state: ProposalState) -> ProposalState:
        """Run a single agent"""
        agent_functions = {
            "Proposal Orchestrator Agent": self.orchestrator_agent,
            "Tech Lead Agent": self.tech_lead_agent,
            "Estimation Agent": self.estimation_agent,
            "Timeline Agent": self.timeline_agent,
            "Legal & Compliance Agent": self.legal_agent,
            "Sales/Marketing Agent": self.sales_agent,
            "Proposal Orchestrator Agent (Final)": self.final_orchestrator_agent
        }
        
        if agent_name in agent_functions:
            return agent_functions[agent_name](state)
        return state

# Simplified system getter
def get_simple_langgraph_system():
    """Get or create simplified LangGraph system"""
    if 'langgraph_system' not in st.session_state:
        config = AzureOpenAIConfig()
        st.session_state.langgraph_system = SimpleLangGraphProposalSystem(config)
    return st.session_state.langgraph_system

def render_manual_langgraph_ui():
    """Manual control version - more reliable"""
    st.title("🤖 Agent Workflow Grid")
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
    
    # Agent status summary (your original beautiful UI)
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
        progress = (completed_count / 7) * 100
        st.metric("Overall Progress", f"{progress:.1f}%")
    
    st.markdown("---")
    
    # Current agent info
    current_agent = st.session_state.workflow_state["current_agent"]
    
    if current_agent != "completed":
        st.info(f"🎯 **Next Agent:** {current_agent}")
    else:
        st.success("🎉 **All Agents Completed!**")
    
    # Control buttons
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
                # Get list of agents to run
                remaining_agents = []
                temp_agent = current_agent
                
                # Simple mapping of agent sequence
                agent_sequence = [
                    "Proposal Orchestrator Agent",
                    "Tech Lead Agent", 
                    "Estimation Agent",
                    "Timeline Agent",
                    "Legal & Compliance Agent",
                    "Sales/Marketing Agent",
                    "Proposal Orchestrator Agent (Final)"
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
                        # Update progress
                        progress = (i / total_agents) * 100
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
                
                # Complete progress
                progress_bar.progress(100)
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
    
    # Your beautiful agent grid display
    st.markdown("### 🤖 Agent Status Grid")
    cols = st.columns(3)
    
    for i, (agent_name, agent_info) in enumerate(st.session_state.agents_workflow.items()):
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
            
            # Progress bar
            if agent_info["status"] == "active":
                st.progress(0, text=f"{agent_name} - Ready to run")
            elif agent_info["status"] == "completed":
                st.progress(100, text="✅ Completed")
            else:
                st.progress(0, text="⏳ Waiting")
            
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
                # Show agent information
                agent_descriptions = {
                    "Proposal Orchestrator Agent": "Project scope overview, risk assessment, success criteria, and stakeholder management plan",
                    "Tech Lead Agent": "Technology stack recommendations, system architecture, security strategy, and development methodology", 
                    "Estimation Agent": "Detailed cost breakdown, effort estimation, resource allocation, and project budget analysis",
                    "Timeline Agent": "Project phases, milestones, task dependencies, and delivery schedule",
                    "Legal & Compliance Agent": "Regulatory compliance analysis, contract terms, risk assessment, and legal recommendations",
                    "Sales/Marketing Agent": "Executive summary, value propositions, competitive advantages, and client benefits",
                    "Proposal Orchestrator Agent (Final)": "Final synthesis of all sections, consistency check, and comprehensive proposal integration"
                }
                
                description = agent_descriptions.get(modal_agent, "Specialized proposal content")
                
                st.info(f"📝 **What this agent will generate:**\n\n{description}")
                
                st.markdown("### 🎯 Agent Responsibilities:")
                
                if modal_agent == "Proposal Orchestrator Agent":
                    st.markdown("""
                    - Analyze RFP requirements and structure
                    - Identify key project components and priorities
                    - Assess potential risks and mitigation strategies
                    - Define success criteria and metrics
                    - Create stakeholder management approach
                    """)
                elif modal_agent == "Tech Lead Agent":
                    st.markdown("""
                    - Recommend optimal technology stack
                    - Design system architecture and patterns
                    - Plan security implementation strategy
                    - Define development methodology and practices
                    - Specify integration points and APIs
                    """)
                elif modal_agent == "Estimation Agent":
                    st.markdown("""
                    - Break down work into detailed components
                    - Estimate effort for each development area
                    - Calculate resource requirements and costs
                    - Factor in risk buffers and contingencies
                    - Provide transparent cost justification
                    """)
                elif modal_agent == "Timeline Agent":
                    st.markdown("""
                    - Define project phases and milestones
                    - Map task dependencies and critical path
                    - Schedule resource allocation over time
                    - Plan review and approval gates
                    - Set realistic delivery expectations
                    """)
                elif modal_agent == "Legal & Compliance Agent":
                    st.markdown("""
                    - Review regulatory compliance requirements
                    - Analyze contract terms and conditions
                    - Assess legal risks and liabilities
                    - Recommend data protection measures
                    - Ensure industry-specific compliance
                    """)
                elif modal_agent == "Sales/Marketing Agent":
                    st.markdown("""
                    - Craft compelling value propositions
                    - Highlight competitive advantages
                    - Create executive summary content
                    - Demonstrate ROI and business benefits
                    - Position solution against requirements
                    """)
                elif modal_agent == "Proposal Orchestrator Agent (Final)":
                    st.markdown("""
                    - Synthesize all agent outputs into cohesive document
                    - Ensure consistency across all sections
                    - Address any gaps or overlaps
                    - Create smooth narrative flow
                    - Finalize executive summary and recommendations
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
    
    # Completion handling
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
        
        if st.button("✅ Review Agent Work", type="primary"):
            st.session_state.step = 'agent_check'
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
    """Create a comprehensive prompt for RFP analysis"""
    return f"""
You are an expert RFP analyst. Analyze the following RFP document and extract key information in a structured format.

RFP Document:
{rfp_text}

Please analyze this RFP and provide a JSON response with the following structure:

{{
    "project_overview": {{
        "title": "Project title",
        "description": "Brief project description",
        "type": "Type of project (e.g., Software Development, Infrastructure, etc.)"
    }},
    "technical_requirements": [
        "List of technical requirements"
    ],
    "functional_requirements": [
        "List of functional requirements"
    ],
    "compliance_requirements": [
        "List of compliance and regulatory requirements"
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
        "Criteria for proposal evaluation"
    ],
    "vendor_requirements": [
        "Requirements for vendors/suppliers"
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
        "Key components that need specialized attention"
    ]
}}

Ensure the response is valid JSON. If any section is not found in the RFP, use an empty array [] or empty string "" as appropriate.
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
                    "content": "You are an expert RFP analyst. Always respond with valid JSON format."
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
    """Render the Azure OpenAI parsing step"""
    st.title("🔍 Parsing RFP Document with Azure OpenAI")
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
            st.warning("⚠️ Azure OpenAI not configured. Using mock parsing for demonstration.")
            
            # Fallback to mock parsing
            st.markdown("""
            <div class="parsing-animation" style="padding: 2rem; border-radius: 10px; text-align: center; color: white; margin: 1rem 0;">
                <h3>🤖 Mock AI Analysis (Azure OpenAI not configured)...</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Simulate progress for demo
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(101):
                progress_bar.progress(i)
                status_text.text(f"Mock parsing progress: {i}%")
                if i % 20 == 0:
                    time.sleep(0.1)
            
            # Use mock data for tutorial
            if st.session_state.tutorial_mode and hasattr(st.session_state, 'identified_components'):
                parsed_data = {
                    "project_overview": {
                        "title": "Sample Project",
                        "description": "Mock analysis results",
                        "type": "Software Development"
                    },
                    "identified_components": st.session_state.identified_components
                }
            else:
                parsed_data = {
                    "project_overview": {
                        "title": "Parsed Project",
                        "description": "Analysis completed with mock data",
                        "type": "Software Development"
                    },
                    "identified_components": [
                        "Technical Requirements", "Project Scope", "Timeline Constraints",
                        "Budget Parameters", "Compliance Requirements", "Deliverables"
                    ]
                }
        else:
            # Real Azure OpenAI parsing
            st.markdown("""
            <div class="parsing-animation" style="padding: 2rem; border-radius: 10px; text-align: center; color: white; margin: 1rem 0;">
                <h3>🧠 Analyzing your document...</h3>
            </div>
            """, unsafe_allow_html=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            details_text = st.empty()
            
            # Parsing steps with Azure OpenAI
            parsing_steps = [
                "Connecting to Azure OpenAI...",
                "Extracting project overview...",
                "Identifying technical requirements...",
                "Analyzing compliance needs...",
                "Extracting timeline information...",
                "Analyzing budget parameters...",
                "Identifying deliverables...",
                "Generating component breakdown..."
            ]
            
            # Show initial progress
            for i in range(30):
                progress_bar.progress(i)
                status_text.text(f"Preparing analysis: {i}%")
                if i % 10 == 0 and i > 0:
                    step_index = min((i // 10) - 1, len(parsing_steps) - 1)
                    details_text.text(f"Step: {parsing_steps[step_index]}")
                time.sleep(0.05)
            
            # Actual Azure OpenAI call
            details_text.text("Calling Azure OpenAI API...")
            parsed_data = parse_rfp_with_azure_openai(st.session_state.rfp_text, config)
            
            if parsed_data:
                # Validate and clean the data
                parsed_data = validate_parsed_data(parsed_data)
                
                # Complete the progress animation
                for i in range(30, 101):
                    progress_bar.progress(i)
                    status_text.text(f"Processing results: {i}%")
                    if i % 15 == 0:
                        step_index = min(4 + (i // 15), len(parsing_steps) - 1)
                        details_text.text(f"Step: {parsing_steps[step_index]}")
                    time.sleep(0.03)
            else:
                st.error("❌ Failed to parse RFP with Azure OpenAI")
                return
        
        st.success("✅ RFP Analysis complete! Document parsed and components identified.")
        
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
            
            st.markdown("#### 🎯 Key Components Identified")
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
        with st.expander("🔍 Detailed Analysis Results"):
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
        
        if st.button("🤖 Dispatch to Agents", type="primary"):
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

# Sample RFP documents for tutorial
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

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 'upload'
if 'tutorial_mode' not in st.session_state:
    st.session_state.tutorial_mode = False
if 'current_agent' not in st.session_state:
    st.session_state.current_agent = None
if 'agents_workflow' not in st.session_state:
    st.session_state.agents_workflow = {
        "Proposal Orchestrator Agent": {
            "task": "Orchestrating proposal generation and RFP breakdown",
            "status": "pending",
            "progress": 0,
            "output": "",
            "details": "Analyzing RFP structure and identifying key components for specialized agents",
            "estimated_time": "2-3 minutes",
            "feedback_requested": False,
            "human_feedback": "",
            "feedback_incorporated": False
        },
        "Tech Lead Agent": {
            "task": "Handling technical requirements and architecture",
            "status": "pending",
            "progress": 0,
            "output": "",
            "details": "Creating technical architecture, system design, and technology stack recommendations",
            "estimated_time": "8-10 minutes",
            "feedback_requested": False,
            "human_feedback": "",
            "feedback_incorporated": False
        },
        "Estimation Agent": {
            "task": "Estimating effort, cost, and resource allocation",
            "status": "pending",
            "progress": 0,
            "output": "",
            "details": "Calculating project costs, resource requirements, and effort estimations",
            "estimated_time": "5-7 minutes",
            "feedback_requested": False,
            "human_feedback": "",
            "feedback_incorporated": False
        },
        "Timeline Agent": {
            "task": "Creating project timeline and milestones",
            "status": "pending",
            "progress": 0,
            "output": "",
            "details": "Developing project schedules, milestones, and dependency mapping",
            "estimated_time": "4-6 minutes",
            "feedback_requested": False,
            "human_feedback": "",
            "feedback_incorporated": False
        },
        "Legal & Compliance Agent": {
            "task": "Reviewing legal clauses and compliance requirements",
            "status": "pending",
            "progress": 0,
            "output": "",
            "details": "Ensuring legal compliance, risk assessment, and contract review",
            "estimated_time": "6-8 minutes",
            "feedback_requested": False,
            "human_feedback": "",
            "feedback_incorporated": False
        },
        "Sales/Marketing Agent": {
            "task": "Creating executive summary and value propositions",
            "status": "pending",
            "progress": 0,
            "output": "",
            "details": "Crafting compelling value propositions and executive summaries",
            "estimated_time": "4-5 minutes",
            "feedback_requested": False,
            "human_feedback": "",
            "feedback_incorporated": False
        },
        "Proposal Orchestrator Agent (Final)": {
            "task": "Synthesizing and refining all proposal sections",
            "status": "pending",
            "progress": 0,
            "output": "",
            "details": "Consolidating all sections and ensuring consistency across the proposal",
            "estimated_time": "3-4 minutes",
            "feedback_requested": False,
            "human_feedback": "",
            "feedback_incorporated": False
        }
    }
if 'consolidated_document' not in st.session_state:
    st.session_state.consolidated_document = ""
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []

# Sidebar navigation
st.sidebar.title("🚀 Proposal Generator")

# Tutorial mode toggle
if st.sidebar.checkbox("🎓 Tutorial Mode", value=st.session_state.tutorial_mode):
    st.session_state.tutorial_mode = True
    if st.session_state.tutorial_mode:
        st.sidebar.markdown("""
        <div class="tutorial-banner">
            <h4>🎓 Tutorial Mode Active</h4>
            <p>Experience the full workflow with sample data</p>
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

# Step 1: Upload RFP document
if st.session_state.step == 'upload':
    st.title("📄 Upload RFP Document")
    
    if st.session_state.tutorial_mode:
        st.markdown("""
        <div class="tutorial-banner">
            <h3>🎓 Tutorial Mode: Sample Documents Available</h3>
            <p>Choose from our sample RFP documents to experience the complete workflow</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("Welcome to the AI-powered Proposal Generator! Start by uploading your RFP document or try our tutorial.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.tutorial_mode:
            st.markdown("### 🎓 Choose a Sample Document")
            selected_sample = st.selectbox(
                "Select a sample RFP to explore:",
                list(SAMPLE_DOCUMENTS.keys()),
                help="These are realistic RFP examples to demonstrate the system"
            )
            
            if selected_sample:
                sample_doc = SAMPLE_DOCUMENTS[selected_sample]
                st.markdown(f"**Sample Document:** {sample_doc['filename']}")
                
                with st.expander("📄 Preview Sample Content"):
                    st.text_area("RFP Content Preview", sample_doc['content'], height=200, disabled=True)
                
                if st.button("🚀 Start Tutorial with Sample", type="primary"):
                    st.session_state.rfp_name = sample_doc['filename']
                    st.session_state.rfp_content = sample_doc['content']
                    st.session_state.identified_components = sample_doc['components']
                    st.session_state.step = 'parsing'
                    st.rerun()
        else:
            uploaded_file = st.file_uploader(
                "Choose an RFP document", 
                type=["pdf", "docx", "txt"],
                help="Supported formats: PDF, DOCX, TXT"
            )
            
            if uploaded_file is not None:
                st.success(f"✅ File uploaded: {uploaded_file.name}")
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
                
                if st.button("🚀 Start Processing", type="primary"):
                    st.session_state.step = 'parsing'
                    st.rerun()
    
    with col2:
        st.markdown("### 📋 What happens next?")
        st.markdown("""
        1. **Document Parsing** - AI analyzes your RFP
        2. **Agent Dispatch** - Specialized agents get to work
        3. **Human Feedback** - Review and guide agent work
        4. **Content Generation** - Each agent creates their section
        5. **Quality Check** - Review and refine outputs
        6. **Final Document** - Export your complete proposal
        """)
        
        if not st.session_state.tutorial_mode:
            st.markdown("---")
            st.markdown("### 🎓 New to the system?")
            if st.button("Try Tutorial Mode"):
                st.session_state.tutorial_mode = True
                st.rerun()
    add_config_help_to_upload()

# Step 2: Parsing simulation
# Step 2: Azure OpenAI Parsing
elif st.session_state.step == 'parsing':
    render_azure_openai_parsing_step()

# Step 3: Agent Grid
elif st.session_state.step == 'agent_grid':
    render_manual_langgraph_ui()


# Step 4: Human Feedback
elif st.session_state.step == 'feedback':
    st.title("💬 Provide Human Feedback")
    st.markdown("Guide the AI agents with your expertise and domain knowledge")
    
    # Find agents requesting feedback
    feedback_agents = {name: info for name, info in st.session_state.agents_workflow.items() 
                      if info["feedback_requested"] and not info["feedback_incorporated"]}
    
    if not feedback_agents:
        st.info("No agents are currently requesting feedback.")
        if st.button("🔙 Back to Agent Grid"):
            st.session_state.step = 'agent_grid'
            st.rerun()
    else:
        selected_feedback_agent = st.selectbox(
            "Select agent to provide feedback:",
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
            
            # Show current work/direction
            st.markdown("### 🔍 Current Work Direction")
            current_work = {
                "Proposal Orchestrator Agent": "Identified 6 main RFP components. Planning to dispatch technical requirements to Tech Lead Agent first, followed by parallel processing of estimation and timeline.",
                "Tech Lead Agent": "Recommending microservices architecture with React frontend and Node.js backend. Considering AWS cloud deployment with containerization.",
                "Estimation Agent": "Initial cost estimate at $85,000 based on 850 development hours. Factoring in 15% risk buffer and resource availability.",
                "Timeline Agent": "Proposing 26-week timeline with 4-week planning phase, 16-week development, and 6-week testing/deployment.",
                "Legal & Compliance Agent": "Reviewing GDPR and SOC2 requirements. Identifying standard contract terms and liability clauses.",
                "Sales/Marketing Agent": "Focusing on scalability and modern technology stack as key value propositions. Emphasizing 10+ years experience.",
                "Proposal Orchestrator Agent (Final)": "Planning to synthesize all sections with focus on consistency and professional presentation."
            }
            
            st.info(current_work.get(selected_feedback_agent, "Working on assigned tasks..."))
            
            # Feedback form
            st.markdown("### 💭 Your Feedback")
            feedback_type = st.radio(
                "Type of feedback:",
                ["Direction Guidance", "Technical Correction", "Business Insight", "Risk Concern", "Additional Requirements"]
            )
            
            feedback_text = st.text_area(
                "Provide your feedback:",
                placeholder="Share your insights, corrections, or additional requirements...",
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
                    
                    st.success(f"✅ Feedback submitted to {selected_feedback_agent}!")
                    time.sleep(1)
                    st.session_state.step = 'agent_grid'
                    st.rerun()
                else:
                    st.error("Please provide feedback before submitting.")
        
        with col2:
            st.markdown("### 📊 Feedback Guidelines")
            st.markdown("""
            **Direction Guidance:**
            - Suggest alternative approaches
            - Recommend specific technologies
            - Guide architectural decisions
            
            **Technical Correction:**
            - Point out technical inaccuracies
            - Suggest better solutions
            - Correct assumptions
            
            **Business Insight:**
            - Share domain knowledge
            - Highlight business priorities
            - Suggest value propositions
            
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

# Step 5: Check Agent Work
elif st.session_state.step == 'agent_check':
    st.title("✅ Review Agent Work")
    st.markdown("Examine the output from each specialized agent and add to your proposal")
    
    # Agent selector
    selected_agent = st.selectbox(
        "Select an agent to review their work:",
        list(st.session_state.agents_workflow.keys()),
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
        
        # Generate mock output if not exists
        if agent_info['output'] == "":
            mock_outputs = {
                "Proposal Orchestrator Agent": """
**RFP Analysis Summary**
- Project Type: Custom Software Development
- Key Requirements: Web application with mobile responsiveness
- Compliance Needs: GDPR, SOC2 Type II
- Budget Range: $50,000 - $100,000
- Timeline: 6-8 months
- Technical Stack: Modern web technologies required
                """,
                "Tech Lead Agent": """
**Technical Architecture Proposal**

**Recommended Technology Stack:**
- Frontend: React.js with TypeScript
- Backend: Node.js with Express.js
- Database: PostgreSQL with Redis caching
- Cloud: AWS with containerized deployment
- Security: OAuth 2.0, JWT tokens, SSL/TLS

**System Architecture:**
- Microservices architecture for scalability
- RESTful API design with GraphQL for complex queries
- Automated CI/CD pipeline with testing
- Monitoring and logging with CloudWatch
                """,
                "Estimation Agent": """
**Project Estimation Breakdown**

**Development Effort:**
- Frontend Development: 320 hours ($32,000)
- Backend Development: 280 hours ($28,000)
- Database Design: 80 hours ($8,000)
- Testing & QA: 120 hours ($12,000)
- DevOps & Deployment: 60 hours ($6,000)

**Total Estimated Cost: $86,000**
**Timeline: 26 weeks (6.5 months)**
**Risk Buffer: 15% ($12,900)**
                """,
                "Timeline Agent": """
**Project Timeline & Milestones**

**Phase 1: Planning & Design (Weeks 1-4)**
- Requirements analysis
- System design
- UI/UX mockups

**Phase 2: Development (Weeks 5-20)**
- Sprint 1-4: Core functionality
- Sprint 5-8: Advanced features
- Sprint 9-12: Integration & testing

**Phase 3: Testing & Deployment (Weeks 21-26)**
- User acceptance testing
- Performance optimization
- Production deployment
- Knowledge transfer
                """,
                "Legal & Compliance Agent": """
**Legal & Compliance Assessment**

**Contract Terms:**
- Intellectual property rights clearly defined
- Data protection clauses included
- Liability limitations standard
- Payment terms: 30% upfront, 40% at milestone, 30% completion

**Compliance Requirements:**
- GDPR compliance for EU users
- SOC2 Type II certification required
- Regular security audits
- Data encryption at rest and in transit

**Risk Assessment: LOW**
                """,
                "Sales/Marketing Agent": """
**Executive Summary & Value Proposition**

**Why Choose Our Solution:**
- 10+ years of experience in custom software development
- Proven track record with 200+ successful projects
- Agile methodology ensuring transparency and flexibility
- 24/7 support and maintenance included

**Key Benefits:**
- Scalable architecture supporting future growth
- Modern, responsive design for optimal user experience
- Robust security measures protecting sensitive data
- Comprehensive testing ensuring reliability

**Competitive Advantage:**
- 40% faster development using our proprietary frameworks
- 99.9% uptime guarantee with our hosting solution
                """,
                "Proposal Orchestrator Agent (Final)": """
**Final Proposal Synthesis**

This comprehensive proposal addresses all requirements outlined in your RFP. Our solution combines cutting-edge technology with proven methodologies to deliver a robust, scalable web application.

**Key Highlights:**
- Complete technical solution with modern architecture
- Realistic timeline with built-in risk mitigation
- Competitive pricing with transparent breakdown
- Full compliance with legal and regulatory requirements

**Next Steps:**
1. Proposal review and feedback
2. Contract negotiation and signing
3. Project kickoff and team introduction
4. Development commencement
                """
            }
            
            # Incorporate feedback into output if available
            base_output = mock_outputs.get(selected_agent, "No output available.")
            if agent_info['human_feedback']:
                base_output += f"\n\n**Note:** This output has been refined based on your feedback: {agent_info['human_feedback'][:100]}..."
            
            agent_info['output'] = base_output
        
        st.markdown("### 📄 Agent Output")
        st.markdown(agent_info['output'])
        
        # Feedback option for completed work
        if agent_info['status'] == 'completed':
            with st.expander("💬 Provide Additional Feedback"):
                additional_feedback = st.text_area(
                    "Any additional feedback for this output?",
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
                st.success(f"✅ Added output from {selected_agent} to consolidated document.")
            else:
                st.warning("This agent's output is already in the consolidated document.")
    
    with col2:
        st.markdown("### 📊 Agent Status")
        st.metric("Progress", f"{agent_info['progress']}%")
        st.metric("Status", agent_info['status'].title())
        
        if agent_info['human_feedback']:
            st.markdown("### 💭 Feedback Given")
            st.success("✅ Human feedback provided")
        
        # Show which outputs are already added
        st.markdown("### 📋 Added to Proposal")
        for agent_name in st.session_state.agents_workflow.keys():
            if f"### {agent_name}" in st.session_state.consolidated_document:
                st.markdown(f"✅ {agent_name}")
            else:
                st.markdown(f"⚪ {agent_name}")
    
    st.markdown("---")
    
    # Check if all outputs are added
    added_count = sum(1 for agent_name in st.session_state.agents_workflow.keys() 
                     if f"### {agent_name}" in st.session_state.consolidated_document)
    
    if added_count == len(st.session_state.agents_workflow):
        st.success("🎉 All agent outputs have been added to the proposal!")
    
    if st.button("📋 Proceed to Consolidate Document", type="primary"):
        st.session_state.step = 'consolidate'
        st.rerun()

# Step 6: Consolidate and Export
elif st.session_state.step == 'consolidate':
    st.title("📋 Consolidate and Export Proposal")
    st.markdown("Review your complete proposal document and export it")
    
    if st.session_state.consolidated_document.strip() == "":
        st.warning("⚠️ No content in consolidated document. Please add outputs from agents first.")
        if st.button("⬅️ Go Back to Review"):
            st.session_state.step = 'agent_check'
            st.rerun()
    else:
        # Document preview
        st.markdown("### 📄 Proposal Document Preview")
        
        # Add header to document
        header = f"""# Proposal Response to RFP: {st.session_state.get('rfp_name', 'Your Project')}

**Generated on:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
**Generated by:** AI Proposal Generator System
**Human Feedback Sessions:** {len(st.session_state.feedback_history)}

---
"""
        
        # Add feedback summary if any
        if st.session_state.feedback_history:
            feedback_summary = "\n## Human Feedback Integration Summary\n"
            feedback_summary += f"This proposal incorporates {len(st.session_state.feedback_history)} human feedback sessions:\n"
            for feedback in st.session_state.feedback_history:
                feedback_summary += f"- {feedback['agent']}: {feedback['type']} ({feedback['priority']} priority)\n"
            feedback_summary += "\n---\n"
        else:
            feedback_summary = ""
        
        full_document = header + feedback_summary + st.session_state.consolidated_document
        
        # Show document in expandable text area
        st.text_area(
            "Complete Proposal Document", 
            value=full_document, 
            height=400,
            help="This is your complete proposal document ready for export"
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
                file_name=f"proposal_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                type="primary"
            )
        
        with col2:
            # Markdown export
            st.download_button(
                label="📝 Download as Markdown",
                data=proposal_bytes,
                file_name=f"proposal_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
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
                file_name=f"feedback_history_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        
        # Workflow completion
        st.success("🎉 Proposal generation completed successfully!")
        
        if st.session_state.tutorial_mode:
            st.info("🎓 Tutorial completed! You've experienced the full workflow with human feedback integration.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Start New Proposal", type="primary"):
                # Reset all session state
                for key in list(st.session_state.keys()):
                    if key not in ['tutorial_mode']:  # Keep tutorial mode setting
                        del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("📧 Share Feedback"):
                st.info("Thank you for using the AI Proposal Generator! Your feedback helps us improve.")

# Sidebar status
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Session Info")
if 'rfp_name' in st.session_state:
    st.sidebar.markdown(f"**RFP:** {st.session_state.rfp_name}")

current_time = datetime.now().strftime("%H:%M:%S")
st.sidebar.markdown(f"**Time:** {current_time}")

if st.session_state.step in ['agent_grid', 'feedback', 'agent_check', 'consolidate']:
    completed_agents = sum(1 for agent in st.session_state.agents_workflow.values() if agent["status"] == "completed")
    total_agents = len(st.session_state.agents_workflow)
    st.sidebar.progress(completed_agents / total_agents, text=f"Agents: {completed_agents}/{total_agents}")

# Feedback status in sidebar
if st.session_state.feedback_history:
    st.sidebar.markdown("### 💭 Feedback Status")
    st.sidebar.metric("Feedback Sessions", len(st.session_state.feedback_history))
    
    feedback_pending = sum(1 for agent in st.session_state.agents_workflow.values() if agent["feedback_requested"] and not agent["feedback_incorporated"])
    if feedback_pending > 0:
        st.sidebar.warning(f"🔔 {feedback_pending} agents need feedback")
