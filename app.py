import streamlit as st
from typing import Dict, Any, List
from datetime import datetime
import json
import matplotlib.pyplot as plt
import pandas as pd
import random  # Used for placeholder data if backend fields are missing
import seaborn as sns

# Assuming your backend workflow and state are importable
# Make sure these paths are correct for your project structure
from rfp_automation.workflow.enhanced_workflow import EnhancedRFPAutomationWorkflow
from rfp_automation.config.settings import get_settings

# from rfp_automation.workflow.state import EnhancedRFPState # For type hinting if you pass state around

# Initialize settings
settings = get_settings()

parsed = {}

# --- Streamlit Session State Initialization ---
if "rfp_run_complete" not in st.session_state:
    st.session_state.rfp_run_complete = False
if "rfp_state_data" not in st.session_state:
    st.session_state.rfp_state_data = {}
if "user_input_val" not in st.session_state:
    st.session_state.user_input_val = ""
if "reviewer_comments_list" not in st.session_state:  # UI-only persistence for comments
    st.session_state.reviewer_comments_list = []
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📊 Overview"
# --- End Session State Initialization ---


# --- Main Streamlit Application ---
def main():
    st.set_page_config(
        page_title="AI-Powered RFP Automation",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
    <style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
    .risk-high { color: #ff4444; font-weight: bold; }
    .risk-medium { color: #ffaa00; font-weight: bold; }
    .risk-low { color: #4CAF50; font-weight: bold; } /* Changed to a clearer green */
    /* Ensure Streamlit's native metric delta color doesn't clash or override if not desired */
    [data-testid="stMetricDelta"] > div { font-size: 0.875rem !important; }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<h1 class="main-header">🚀 AI-Powered RFP Automation</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Transform vague requirements into comprehensive RFPs with market intelligence and vendor analysis."
    )

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        tavily_api_key_present = bool(
            settings.TAVILY_API_KEY
            and "YOUR_TAVILY_API_KEY" not in settings.TAVILY_API_KEY
        )
        google_api_key_present = bool(
            settings.GOOGLE_API_KEY
            and "YOUR_GOOGLE_API_KEY" not in settings.GOOGLE_API_KEY
        )

        st.subheader("API Status")
        st.caption(
            f"Tavily API Key: {'✅ Present' if tavily_api_key_present else '❌ Missing/Default'}"
        )
        st.caption(
            f"Google API Key: {'✅ Present' if google_api_key_present else '❌ Missing/Default'}"
        )

        st.subheader("Workflow Control")
        # TODO: These checkboxes need backend integration to actually control workflow paths.
        # Currently, the backend 'EnhancedRFPAutomationWorkflow' runs all main steps.
        # Options:
        # 1. Pass flags to workflow.run() and have agents or graph logic react.
        # 2. Select different pre-compiled LangGraph graphs.
        st.checkbox(
            "Enable Market Research (Backend Default: On)",
            value=True,
            disabled=True,
            help="Backend currently always runs this.",
        )
        st.checkbox(
            "Enable Vendor Intelligence (Backend Default: On)",
            value=True,
            disabled=True,
            help="Backend currently always runs this.",
        )
        st.checkbox(
            "Show Smart Suggestions in UI",
            value=True,
            key="show_suggestions_checkbox",
            help="Controls UI display of suggestions.",
        )

        if st.button("🔄 Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]  # Clears all session state
            st.success("Session reset successfully!")
            st.rerun()
    # --- End Sidebar ---

    # --- Main Input Section ---
    st.header("📝 Project Requirements Input")
    sample_inputs = {
        "Logistics Platform": "We need a vendor to build a mobile app for logistics tracking, scalable to 100k users with real-time GPS tracking and analytics dashboard. Our timeline is somewhat urgent.",
        "Healthcare System": "Looking for a healthcare management platform for 50k patients with HIPAA compliance, telemedicine features, and mobile access for enterprise use.",
    }
    input_col1, input_col2 = st.columns([3, 1])
    with input_col1:
        user_input = st.text_area(
            "Enter your project requirements:",
            height=120,
            placeholder="Describe your project needs...",
            value=st.session_state.user_input_val,
            key="main_user_input_area",
        )
        st.session_state.user_input_val = user_input  # Keep it updated

    with input_col2:
        st.write("**Quick Start Examples:**")
        for name, sample in sample_inputs.items():
            if st.button(f"📋 {name}", key=f"sample_{name}"):
                st.session_state.user_input_val = sample
                st.rerun()

    if st.button("🚀 Generate RFP with Market Intelligence", type="primary"):
        if not st.session_state.user_input_val.strip():  # type:ignore
            st.error("Please enter project requirements or select a sample.")
        elif not tavily_api_key_present or not google_api_key_present:
            st.error(
                "API Keys are missing or default. Please configure them in your .env file."
            )
        else:
            st.session_state.rfp_run_complete = False
            st.session_state.rfp_state_data = {}
            st.session_state.reviewer_comments_list = []

            with st.spinner(
                "🔄 Processing requirements via AI agents... This may take a moment."
            ):
                try:
                    workflow = EnhancedRFPAutomationWorkflow(
                        tavily_api_key=settings.TAVILY_API_KEY,
                        chroma_persist_dir=settings.CHROMA_PERSIST_DIRECTORY,
                    )
                    backend_output_state = workflow.run(
                        st.session_state.user_input_val or ""
                    )
                    st.session_state.rfp_state_data = backend_output_state

                    with open("state.json", "w") as f:
                        json.dump(backend_output_state, f)

                    st.session_state.rfp_run_complete = True
                    st.success("✅ RFP generation completed successfully!")
                except Exception as e:
                    st.error(f"An error occurred during RFP generation: {str(e)}")
                    st.exception(e)  # For detailed traceback in console/UI
                    st.session_state.rfp_run_complete = False
    # --- End Main Input Section ---

    # --- Display Results ---
    if st.session_state.rfp_run_complete and st.session_state.rfp_state_data:
        show_suggestions_ui = st.session_state.get("show_suggestions_checkbox", True)
        display_results_tabs(
            st.session_state.rfp_state_data, show_suggestions_ui  # type:ignore
        )  # type:ignore
    elif st.session_state.rfp_run_complete and not st.session_state.rfp_state_data:
        st.warning(
            "Processing seemed complete, but no data was returned from the backend."
        )
    # --- End Display Results ---


# --- Tabbed Display Functions ---
def display_results_tabs(state: Dict[str, Any], show_suggestions: bool):
    tab_names = [
        "📊 Overview",
        "📄 RFP Document",
        "🏢 Vendor Analysis",
        "💡 Insights",
        "📈 Analytics",
    ]
    tabs = st.tabs(tab_names)

    with tabs[0]:  # Overview
        display_overview_tab(state, show_suggestions)
    with tabs[1]:  # RFP Document
        display_rfp_tab(state)
    with tabs[2]:  # Vendor Analysis
        display_vendor_tab(state)
    with tabs[3]:  # Insights
        display_insights_tab(state)
    with tabs[4]:  # Analytics
        display_analytics_tab(state)


def display_overview_tab(state: Dict[str, Any], show_suggestions: bool):

    global parsed

    st.header("📊 Project Overview")
    parsed = state.get("parsed_requirements", {})
    budget = state.get("budget_estimate", {})
    scale_val = parsed.get("scale", 0)

    cols = st.columns(4)
    cols[0].metric("Target Scale", f"{scale_val:,} users" if scale_val else "N/A")

    dev_cost = budget.get("development_cost", 0)
    # TODO Backend (EnhancedBudgetAgent): Calculate and return `cost_per_user_first_year`.
    # Mock: `budget.get('cost_per_user_first_year', 0)`
    cost_per_user_placeholder = (
        (budget.get("total_first_year", dev_cost) / scale_val) if scale_val > 0 else 0
    )
    cost_per_user_actual = budget.get(
        "cost_per_user_first_year", cost_per_user_placeholder
    )
    cols[1].metric(
        "Est. Dev Cost",
        f"${dev_cost:,.0f}" if dev_cost else "N/A",
        delta=(
            f"${cost_per_user_actual:.2f}/user/yr (Total Y1)"
            if cost_per_user_actual
            else "Cost/user N/A"
        ),
    )

    cols[2].metric("Domain", parsed.get("domain", "N/A").title())
    platforms = parsed.get("platform", [])
    cols[3].metric(
        "Platforms",
        f"{len(platforms)} platform(s)" if platforms else "N/A",
        delta=", ".join(platforms) if platforms else None,
    )

    st.subheader("🎯 Parsed Requirements")
    req_cols = st.columns(2)
    req_cols[0].write(f"**Domain:** {parsed.get('domain', 'N/A').title()}")
    req_cols[0].write(f"**Scale:** {parsed.get('scale', 0):,} users")
    req_cols[0].write(f"**Platforms:** {', '.join(parsed.get('platform', ['N/A']))}")
    # TODO Backend (NLPParserAgent): Extract and return `urgency` in `parsed_requirements`.
    req_cols[0].write(f"**Urgency:** {parsed.get('urgency', '')}")

    req_cols[1].write("**Features Identified:**")
    features = parsed.get("features", [])
    if features:
        for feature in features:
            req_cols[1].write(f"• {feature.title()}")
    else:
        req_cols[1].caption("Standard functionality or N/A")

    if show_suggestions:
        suggestions = state.get("suggestions", [])
        if suggestions:
            st.subheader("💡 Smart Suggestions")
            # TODO Backend (SuggestionAgent): Enhance to provide more contextual (domain, market-driven) suggestions.
            # UI will display whatever backend provides.
            for suggestion in suggestions:
                st.info(f"🤖 {suggestion}")
        elif parsed:  # only show if parsing happened
            st.info(
                "No specific suggestions currently generated by the backend for this input."
            )


def display_rfp_tab(state: Dict[str, Any]):
    st.header("📄 Generated RFP Document")
    rfp_doc = state.get("rfp_document", "")

    global parsed

    if rfp_doc:
        # TODO Backend (RFPGeneratorAgent): Enhance template for more comprehensive RFP (ID, dates, all sections from mock).
        st.markdown(rfp_doc)
        st.subheader("📤 Export Options")
        exp_cols = st.columns(3)
        rfp_fn = f"RFP_{parsed.get('domain','project')}_{datetime.now().strftime('%Y%m%d')}.md"
        exp_cols[0].download_button(
            "📄 Download as Markdown", rfp_doc, file_name=rfp_fn, mime="text/markdown"
        )
        # TODO: Implement PDF export (e.g., using FPDF2 or ReportLab, would be a new backend utility)
        exp_cols[1].button("📰 Download as PDF", disabled=True)
        # TODO: Implement Email functionality (requires email service integration).
        exp_cols[2].button("📧 Email RFP", disabled=True)

        st.subheader("✏️ Review & Comments (UI Session Only)")
        comment = st.text_area("Add review comments:", key="rfp_review_comment_input")
        if st.button("💾 Save Comment"):
            if comment:
                st.session_state.reviewer_comments_list.append(
                    {"ts": datetime.now().isoformat(), "comment": comment}
                )
                st.success("Comment saved for this session!")
        if st.session_state.reviewer_comments_list:
            st.write("**Previous Comments:**")
            for entry in reversed(st.session_state.reviewer_comments_list):
                st.caption(
                    f"_{datetime.fromisoformat(entry['ts']).strftime('%Y-%m-%d %H:%M')}_: {entry['comment']}"
                )
    else:
        st.warning("No RFP document generated or available.")


def display_vendor_tab(state: Dict[str, Any]):
    st.header("🏢 Vendor Analysis & Recommendations")

    # Mock data for demonstration
    mock_proposals = [
        {
            "vendor_name": "TechNova Solutions",
            "vendor_type": "Enterprise",
            "final_score": 92.5,
            "cost": 250000,
            "timeline_months": 6,
            "risk_level": "Low",
            "risk_score": 0.15,
            "team_size": 12,
            "experience_years": 8,
            "features": ["Cloud Integration", "24/7 Support", "Custom Analytics"],
            "strengths": [
                "Proven track record",
                "Scalable solutions",
                "Strong customer support",
            ],
            "risk_factors": ["Dependency on third-party APIs"],
            "score_breakdown": {
                "Technical Fit": 35,
                "Cost Efficiency": 30,
                "Timeline": 15,
                "Risk Assessment": 10,
                "Team Expertise": 10,
            },
            "risk_analysis": [
                {
                    "Risk": "Data Breach",
                    "Severity": "High",
                    "Likelihood": "Medium",
                    "Impact": "High",
                },
                {
                    "Risk": "Downtime",
                    "Severity": "Medium",
                    "Likelihood": "High",
                    "Impact": "High",
                },
                {
                    "Risk": "Compliance Violation",
                    "Severity": "Low",
                    "Likelihood": "Medium",
                    "Impact": "Medium",
                },
            ],
        },
        {
            "vendor_name": "InnoTech Corp",
            "vendor_type": "Startup",
            "final_score": 85.0,
            "cost": 180000,
            "timeline_months": 5,
            "risk_level": "Medium",
            "risk_score": 0.35,
            "team_size": 8,
            "experience_years": 3,
            "features": ["AI-Powered Tools", "Mobile Compatibility"],
            "strengths": ["Innovative solutions", "Agile development"],
            "risk_factors": ["Limited market presence", "Smaller team size"],
            "score_breakdown": {
                "Technical Fit": 30,
                "Cost Efficiency": 25,
                "Timeline": 20,
                "Risk Assessment": 15,
                "Team Expertise": 10,
            },
            "risk_analysis": [
                {
                    "Risk": "Data Breach",
                    "Severity": "High",
                    "Likelihood": "Medium",
                    "Impact": "High",
                },
                {
                    "Risk": "Downtime",
                    "Severity": "Medium",
                    "Likelihood": "High",
                    "Impact": "High",
                },
                {
                    "Risk": "Compliance Violation",
                    "Severity": "Low",
                    "Likelihood": "Medium",
                    "Impact": "Medium",
                },
            ],
        },
    ]

    mock_recommendations = {
        "ranked_proposals": mock_proposals,
        "top_choice": mock_proposals[0],
        "summary": (
            "After evaluating all proposals, **TechNova Solutions** stands out with a high final score of 92.5. "
            "Their extensive experience, comprehensive feature set, and low-risk profile make them the top recommendation."
        ),
        "decision_matrix": {
            "TechNova Solutions": {
                "Technical Fit": 35,
                "Cost Efficiency": 30,
                "Timeline": 15,
                "Risk Assessment": 10,
                "Team Expertise": 10,
            },
            "InnoTech Corp": {
                "Technical Fit": 30,
                "Cost Efficiency": 25,
                "Timeline": 20,
                "Risk Assessment": 15,
                "Team Expertise": 10,
            },
        },
    }

    # Use mock data for display
    recommendations_data = mock_recommendations
    proposals = recommendations_data.get("ranked_proposals", [])

    if not proposals:
        st.warning("No vendor proposals generated/analyzed.")
        return

    top_choice = recommendations_data.get("top_choice")
    if top_choice:
        st.subheader("🏆 Recommended Vendor")
        tc_cols = st.columns(3)
        tc_cols[0].metric("Vendor", top_choice.get("vendor_name", "N/A"))
        tc_cols[0].metric("Final Score", f"{top_choice.get('final_score', 0):.1f}/100")
        tc_cols[1].metric("Total Cost", f"${top_choice.get('cost', 0):,.2f}")
        tc_cols[1].metric("Timeline", f"{top_choice.get('timeline_months', 0)} months")
        risk_level = top_choice.get("risk_level", "N/A")
        risk_class = f"risk-{risk_level.lower()}" if risk_level != "N/A" else ""
        tc_cols[2].markdown(
            f"**Risk Level:** <span class='{risk_class}'>{risk_level}</span>",
            unsafe_allow_html=True,
        )
        tc_cols[2].metric("Team Size", f"{top_choice.get('team_size', 'N/A')} people")

        st.markdown(recommendations_data.get("summary", "Summary not available."))
    elif proposals:
        st.info(
            "Top recommendation details not fully available. Displaying ranked proposals."
        )

    st.subheader("📊 Vendor Comparison Table")
    comp_data = [
        {
            "Rank": i + 1,
            "Vendor": p.get("vendor_name", f"Vendor {i+1}"),
            "Type": p.get("vendor_type", "N/A"),
            "Score": f"{p.get('final_score', p.get('risk_score',0)):.1f}",
            "Cost": f"${p.get('cost', 0):,.0f}",
            "Timeline": f"{p.get('timeline_months', 0)}m",
            "Risk": p.get("risk_level", "N/A"),
            "Experience": f"{p.get('experience_years', 'N/A')}y",
        }
        for i, p in enumerate(proposals)
    ]
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    st.subheader("📋 Detailed Proposals")
    for i, p_data in enumerate(proposals):
        score_disp = (
            f"(Score: {p_data.get('final_score', 'N/A'):.1f})"
            if "final_score" in p_data
            else f"(Risk: {p_data.get('risk_score', 'N/A'):.2f})"
        )
        with st.expander(
            f"#{i+1} - {p_data.get('vendor_name', f'Vendor {i+1}')} {score_disp}"
        ):
            st.write(
                f"**Cost:** ${p_data.get('cost',0):,.2f} | **Timeline:** {p_data.get('timeline_months',0)} months"
            )
            st.write(
                f"**Type:** {p_data.get('vendor_type','N/A')} | **Team:** {p_data.get('team_size','N/A')} | **Experience:** {p_data.get('experience_years','N/A')} yrs"
            )
            risk_level_p = p_data.get("risk_level", "N/A")
            risk_class_p = (
                f"risk-{risk_level_p.lower()}" if risk_level_p != "N/A" else ""
            )
            st.markdown(
                f"**Risk Level:** <span class='{risk_class_p}'>{risk_level_p}</span> (Score: {p_data.get('risk_score', -1):.2f})",
                unsafe_allow_html=True,
            )

            prop_cols = st.columns(2)
            prop_cols[0].write("**Features Offered:**")
            for feat in p_data.get("features", ["N/A"]):
                prop_cols[0].caption(f"• {feat}")
            prop_cols[1].write("**Strengths:**")
            for strength in p_data.get("strengths", ["N/A"]):
                prop_cols[1].caption(f"• {strength}")

            if p_data.get("risk_factors"):
                st.write("**Vendor-Identified Risk Factors:**")
                for risk in p_data.get("risk_factors", []):
                    st.caption(f"• {risk}")
            if p_data.get("score_breakdown"):
                st.write("**Score Breakdown:**")
                score_breakdown = p_data.get("score_breakdown")
                # Create columns to display metrics side by side
                cols = st.columns(len(score_breakdown))

                for i, (category, score) in enumerate(score_breakdown.items()):
                    with cols[i]:
                        st.metric(label=category, value=f"{score}%", delta=None)
            if p_data.get("risk_analysis"):
                st.write("**Detailed Risk Analysis:**")
                risk_analysis = p_data.get("risk_analysis")
                df_risk = pd.DataFrame(risk_analysis)
                st.dataframe(df_risk)

    if recommendations_data.get("decision_matrix"):
        st.subheader("Decision Matrix")
        # Mock data for demonstration
        decision_matrix = {
            "Youngsoft": {
                "Technical Fit": 35,
                "Cost Efficiency": 30,
                "Timeline": 15,
                "Risk Assessment": 10,
                "Team Expertise": 10,
            },
            "Abysz": {
                "Technical Fit": 30,
                "Cost Efficiency": 25,
                "Timeline": 20,
                "Risk Assessment": 15,
                "Team Expertise": 10,
            },
            "AI Deva": {
                "Technical Fit": 32,
                "Cost Efficiency": 28,
                "Timeline": 18,
                "Risk Assessment": 10,
                "Team Expertise": 10,
            },
        }

        # Convert the decision matrix to a DataFrame
        df_decision = pd.DataFrame(decision_matrix).T

        # Display the decision matrix as a heatmap
        st.subheader("📊 Decision Matrix Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            df_decision, annot=True, cmap="YlGnBu", fmt=".0f", linewidths=0.5, ax=ax
        )
        ax.set_title("Vendor Decision Matrix Scores")
        st.pyplot(fig)


def display_insights_tab(state: Dict[str, Any]):
    st.header("💡 System Insights & Recommendations")
    market_research = state.get("market_research", {})
    security = state.get("security_requirements", {})
    tech = state.get("tech_recommendations", {})
    budget = state.get("budget_estimate", {})
    cached_knowledge = state.get("cached_knowledge", [])

    if market_research:
        st.subheader("📈 Market Research Insights")
        st.metric(
            "Sources Found (Tavily)",
            state.get("search_metadata", {}).get("sources_found", 0),
        )

        st.write("**Market Trends:**")
        for trend in market_research.get("market_trends", ["N/A"]):
            st.info(trend)

        st.write("**Vendor Landscape Snippets:**")
        for snippet in market_research.get("vendor_landscape", ["N/A"]):
            st.caption(f"• {snippet}")

        if ca := market_research.get("cost_analysis"):
            st.write("**Market Cost Analysis:**")
            # Display metrics
            col1, col2, col3 = st.columns(3)

            col1.metric(
                label="Average Cost",
                value=f"${ca['average_vendor_cost']:,.2f}",
                delta=None,
            )

            col2.metric(
                label="High Estimate",
                value=f"${ca['cost_range']['max']:,.2f}",
                delta=None,
            )

            col3.metric(
                label="Low Estimate",
                value=f"${ca['cost_range']['min']:,.2f}",
                delta=None,
            )

            # Display notes as a list
            if notes := ca.get("notes"):
                st.write("**Notes:**")
                for note in notes:
                    st.caption(f"• {note}")

    if cached_knowledge:
        st.subheader("🧠 Knowledge Base (ChromaDB)")
        st.write(f"Retrieved {len(cached_knowledge)} relevant cached insights.")
        for i, k in enumerate(cached_knowledge[:3]):
            with st.expander(
                f"Insight {i+1} (Score: {k.get('relevance_score', 0):.2f}) - Source: {k.get('source', 'N/A')}"
            ):
                st.caption(k.get("content", "N/A")[:500] + "...")

    st.subheader("🔒 Security & Compliance")
    st.write("**Compliance Suggested:**")
    if security:
        st.subheader("🔒 Security & Compliance")

        # Display compliance requirements as a list
        compliance_list = security.get("compliance", [])
        st.write("**Compliance Suggested:**")
        for item in compliance_list:
            st.caption(f"• {item}")

        # Display additional security details
        extra_security = {k: v for k, v in security.items() if k != "compliance"}
        if extra_security:
            st.write("**Additional Security Details:**")
            for k, v in extra_security.items():
                st.markdown(f"- **{k.replace('_', ' ').title()}**: {v}")

    if extra := {k: v for k, v in security.items() if k != "compliance"}:
        st.write("**Additional Security Details:**")
        for k, v in extra.items():
            st.markdown(f"- **{k.replace('_', ' ').title()}**: {v}")

    if tech:
        st.subheader("⚙️ Technology Stack Recommendations")

        # Display technology recommendations
        for k, v in tech.items():
            st.markdown(f"- **{k.replace('_', ' ').title()}**: {v}")

    st.subheader("💰 Budget Analysis")

    if budget:
        # Create columns for displaying metrics side by side
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Est. Development Cost",
                value=f"${budget['development_cost']:,.2f}",
            )

        with col2:
            st.metric(
                label="Est. Total First Year",
                value=f"${budget['total_first_year']:,.2f}",
            )

        with col3:
            st.metric(
                label="Market Adjustment Factor",
                value=f"{budget['market_adjustment_factor']:.2f}x",
            )

        if cb := budget.get("cost_breakdown"):
            st.write("**Cost Breakdown:**")
            for k, v in cb.items():
                st.markdown(f"- **{k}**: {v}%")
    else:
        st.warning("Budget details not available.")


def display_analytics_tab(state: Dict[str, Any]):
    st.header("📈 Analytics & Performance")
    audit_log = state.get("audit_log", [])
    market_research = state.get("market_research", {})
    search_meta = state.get("search_metadata", {})
    proposals = state.get("proposal_scores", [])

    st.subheader("⚡ Processing Performance")
    if audit_log:
        st.metric("Workflow Steps Logged (Audit)", len(audit_log))
        # TODO: More detailed processing metrics (e.g., agent timings) would require backend changes.
    else:
        st.caption("Audit log not available.")

    st.subheader("🔍 Market Intelligence Metrics")
    if market_research or search_meta:
        # TODO Backend (MarketResearchAgent): Provide `sources_analyzed` (count).
        st.metric("Tavily Sources Found", search_meta.get("sources_found", 0))
        st.metric("Tavily Searches Performed", search_meta.get("total_searches", 0))
        st.metric(
            "Market Trends Identified", len(market_research.get("market_trends", []))
        )
    else:
        st.caption("Market intelligence metrics not available.")

    st.subheader("🏢 Vendor Analysis Metrics")
    if proposals:
        # TODO Backend (RecommendationEngine): Ensure `final_score` in proposals.
        # TODO Backend (RiskEvaluatorAgent): Ensure `risk_level` (string) in proposals.
        # TODO Backend (VendorIntelligenceAgent): Ensure `cost` in proposals.
        st.metric("Proposals Analyzed", len(proposals))
        valid_scores_an = [
            p.get("final_score", p.get("risk_score"))
            for p in proposals
            if p.get("final_score", p.get("risk_score")) is not None
        ]
        avg_score_an = (
            sum(valid_scores_an) / len(valid_scores_an) if valid_scores_an else 0
        )
        st.metric("Average Score/Risk", f"{avg_score_an:.1f}")
        high_risk_an = sum(
            1 for p in proposals if p.get("risk_level") == "High"
        )  # Relies on TODO
        st.metric("High Risk Proposals", high_risk_an)

        st.subheader("📊 Vendor Comparison Charts")
        # TODO: Populate charts with actual data once backend provides all necessary fields consistently.
        # (final_score, cost, vendor_name, risk_level string, timeline_months)
        if len(proposals) > 1:
            try:
                scores_c = [
                    p.get("final_score", random.uniform(40, 90)) for p in proposals
                ]
                costs_c = [
                    p.get("cost", random.randint(50000, 200000)) for p in proposals
                ]
                vendors_c = [
                    p.get("vendor_name", f"Vendor {i+1}")
                    for i, p in enumerate(proposals)
                ]
                risk_levels_c = [
                    p.get("risk_level", random.choice(["Low", "Medium", "High"]))
                    for p in proposals
                ]

                fig_an_score_cost, ax_an_sc = plt.subplots()
                ax_an_sc.scatter(costs_c, scores_c, alpha=0.7)
                for i, txt in enumerate(vendors_c):
                    ax_an_sc.annotate(txt, (costs_c[i], scores_c[i]))
                ax_an_sc.set_xlabel("Cost ($)")
                ax_an_sc.set_ylabel("Final Score (Placeholder)")
                ax_an_sc.set_title("Score vs Cost")
                st.pyplot(fig_an_score_cost)
                plt.close(fig_an_score_cost)

                risk_counts_c = pd.Series(risk_levels_c).value_counts()
                fig_an_risk, ax_an_r = plt.subplots()
                ax_an_r.bar(
                    risk_counts_c.index,
                    list(risk_counts_c.values),
                    color=["green", "orange", "red"][: len(risk_counts_c)],
                )
                ax_an_r.set_title("Risk Level Distribution (Placeholder)")
                ax_an_r.set_xlabel("Risk Level")
                st.pyplot(fig_an_risk)
                plt.close(fig_an_risk)
            except Exception as e_chart:
                st.caption(f"Could not generate charts: {e_chart}")
        else:
            st.caption("Need at least 2 proposals to generate comparison charts.")
    else:
        st.caption("Vendor analysis metrics not available.")

    if audit_log:
        st.subheader("📋 Processing Audit Log (Detailed)")
        with st.expander("View Full Audit Log"):
            for i, entry in enumerate(audit_log):
                st.write(
                    f"**{i+1}. Agent: {entry.get('agent','N/A')} | Action: {entry.get('action','N/A')}**"
                )
                st.caption(f"Timestamp: {entry.get('timestamp','N/A')}")
                if entry.get("data"):
                    st.json(entry.get("data"), expanded=False)
                st.divider()
    else:
        st.caption("Detailed audit log N/A.")

    # TODO: Implement actual analytics data collection and JSON export.
    st.subheader("📤 Export Analytics (Placeholder)")
    st.button("📊 Export Analytics Data", disabled=True)


# --- Footer ---
def display_footer():
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚀 <strong>AI-Powered Multi-Agent RFP Automation System</strong></p>
        <p>Utilizing LangGraph, Tavily, ChromaDB, and Streamlit</p>
        <p><em>This UI aims to integrate with the backend. Some features depend on backend enhancements.</em></p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# --- Run Application ---
if __name__ == "__main__":
    main()
    display_footer()
