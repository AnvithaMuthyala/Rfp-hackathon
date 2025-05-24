import streamlit as st
from typing import Dict, Any, List
from datetime import datetime
import json
import matplotlib.pyplot as plt
import pandas as pd
import random  # Used for placeholder data if backend fields are missing

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
        '<h1 class="main-header">🚀 AI-Powered RFP Automation (Integrated)</h1>',
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

        st.subheader("Workflow Control (Conceptual)")
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
            else "Cost/user N/A (TODO)"
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
    req_cols[0].write(f"**Urgency:** {parsed.get('urgency', 'N/A (TODO: NLPParser)')}")

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
        exp_cols[1].button("📰 Download as PDF (TODO)", disabled=True)
        # TODO: Implement Email functionality (requires email service integration).
        exp_cols[2].button("📧 Email RFP (TODO)", disabled=True)

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
    # Backend state: `vendor_proposals` (raw), `proposal_scores` (scored), `recommendations` (ranked, top_choice)
    recommendations_data = state.get("recommendations", {})
    proposals = recommendations_data.get(
        "ranked_proposals", state.get("proposal_scores", [])
    )  # Use best available list

    if not proposals:
        st.warning("No vendor proposals generated/analyzed.")
        # TODO Backend (VendorIntelligenceAgent, RiskEvaluatorAgent, RecommendationEngine): Ensure these agents run and populate their respective state keys.
        return

    top_choice = recommendations_data.get("top_choice")
    if top_choice:
        st.subheader("🏆 Recommended Vendor")
        # TODO Backend (RecommendationEngine): Ensure `top_choice` has all fields: 'vendor_name', 'final_score', 'cost', 'timeline_months', 'risk_level', 'team_size', 'strengths'.
        # TODO Backend (RiskEvaluatorAgent): Provide string `risk_level` in proposals.
        # TODO Backend (VendorIntelligenceAgent): Provide `team_size` in proposals.
        tc_cols = st.columns(3)
        tc_cols[0].metric("Vendor", top_choice.get("vendor_name", "N/A"))
        tc_cols[0].metric("Final Score", f"{top_choice.get('final_score', 0):.1f}/100")
        tc_cols[1].metric("Total Cost", f"${top_choice.get('cost', 0):,.2f}")
        tc_cols[1].metric("Timeline", f"{top_choice.get('timeline_months', 0)} months")
        risk_level = top_choice.get("risk_level", "N/A (TODO)")
        risk_class = f"risk-{risk_level.lower()}" if risk_level != "N/A (TODO)" else ""
        tc_cols[2].markdown(
            f"**Risk Level:** <span class='{risk_class}'>{risk_level}</span>",
            unsafe_allow_html=True,
        )
        tc_cols[2].metric(
            "Team Size", f"{top_choice.get('team_size', 'N/A (TODO)')} people"
        )

        # TODO Backend (RecommendationEngine): Generate `summary` text for top choice.
        st.markdown(
            recommendations_data.get(
                "summary", "Summary N/A (TODO: RecommendationEngine)"
            )
        )
    elif proposals:
        st.info(
            "Top recommendation details not fully available. Displaying ranked proposals."
        )

    st.subheader("📊 Vendor Comparison Table")
    comp_data = [
        {
            "Rank": i + 1,
            "Vendor": p.get("vendor_name", f"Vendor {i+1}"),
            # TODO Backend (VendorIntelligenceAgent): Add `vendor_type`.
            "Type": p.get("vendor_type", "N/A (TODO)"),
            "Score": f"{p.get('final_score', p.get('risk_score',0)):.1f}",  # Fallback
            "Cost": f"${p.get('cost', 0):,.0f}",
            "Timeline": f"{p.get('timeline_months', 0)}m",
            # TODO Backend (RiskEvaluatorAgent): Add string `risk_level`.
            "Risk": p.get("risk_level", "N/A (TODO)"),
            # TODO Backend (VendorIntelligenceAgent): Add `experience_years`.
            "Experience": f"{p.get('experience_years', 'N/A (TODO)')}y",
        }
        for i, p in enumerate(proposals)
    ]
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    st.subheader("📋 Detailed Proposals")
    for i, p_data in enumerate(proposals):
        # TODO Backend (RecommendationEngine): Ensure `final_score` is in proposal.
        score_disp = (
            f"(Score: {p_data.get('final_score', 'N/A'):.1f})"
            if "final_score" in p_data
            else f"(Risk: {p_data.get('risk_score', 'N/A'):.2f})"
        )
        with st.expander(
            f"#{i+1} - {p_data.get('vendor_name', f'Vendor {i+1}')} {score_disp}"
        ):
            # TODO Backend (VendorIntelligenceAgent): Ensure all these fields are in proposal data:
            # `cost`, `timeline_months`, `vendor_type`, `team_size`, `experience_years`, `similar_projects`,
            # `features` (already there), `strengths`, `risk_factors` (vendor identified).
            # TODO Backend (RiskEvaluatorAgent): Ensure `risk_level` (string), `risk_score` (numeric), and `risk_analysis` (dict) are there.
            # TODO Backend (RecommendationEngine): Ensure `score_breakdown` (dict) is there.
            st.write(
                f"**Cost:** ${p_data.get('cost',0):,.2f} | **Timeline:** {p_data.get('timeline_months',0)} months"
            )
            st.write(
                f"**Type:** {p_data.get('vendor_type','N/A (TODO)')} | **Team:** {p_data.get('team_size','N/A (TODO)')} | **Experience:** {p_data.get('experience_years','N/A (TODO)')} yrs"
            )
            risk_level_p = p_data.get("risk_level", "N/A (TODO)")
            risk_class_p = (
                f"risk-{risk_level_p.lower()}" if risk_level_p != "N/A (TODO)" else ""
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
            for strength in p_data.get("strengths", ["N/A (TODO)"]):
                prop_cols[1].caption(f"• {strength}")

            if p_data.get("risk_factors"):
                st.write("**Vendor-Identified Risk Factors:**")
                for risk in p_data.get("risk_factors", []):
                    st.caption(f"• {risk}")
            if p_data.get("score_breakdown"):
                st.write("**Score Breakdown (TODO: RecommendationEngine for data):**")
                st.json(p_data["score_breakdown"], expanded=False)
            if p_data.get("risk_analysis"):  # from mock RiskEvaluator
                st.write(
                    "**Detailed Risk Analysis (TODO: RiskEvaluatorAgent for data):**"
                )
                st.json(p_data["risk_analysis"], expanded=False)

    # TODO Backend (RecommendationEngine): Generate `decision_matrix`.
    if recommendations_data.get("decision_matrix"):
        st.subheader("Decision Matrix (TODO: RecommendationEngine for data)")
        st.json(recommendations_data["decision_matrix"], expanded=False)  # Placeholder


def display_insights_tab(state: Dict[str, Any]):
    st.header("💡 System Insights & Recommendations")
    market_research = state.get("market_research", {})
    security = state.get("security_requirements", {})
    tech = state.get("tech_recommendations", {})
    budget = state.get("budget_estimate", {})
    cached_knowledge = state.get(
        "cached_knowledge", []
    )  # Existing backend provides this

    if market_research:
        st.subheader("📈 Market Research Insights")
        # TODO Backend (MarketResearchAgent): Provide `sources_analyzed` (count) and `cost_analysis` (dict).
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
        if market_research.get("cost_analysis"):
            st.write("**Market Cost Analysis (TODO: MarketResearchAgent for data):**")
            st.json(market_research["cost_analysis"], expanded=False)

    if cached_knowledge:
        st.subheader("🧠 Knowledge Base (ChromaDB)")
        st.write(f"Retrieved {len(cached_knowledge)} relevant cached insights.")
        for i, k in enumerate(cached_knowledge[:3]):
            with st.expander(
                f"Insight {i+1} (Score: {k.get('relevance_score',0):.2f}) - Source: {k.get('source','N/A')}"
            ):
                st.caption(k.get("content", "N/A")[:500] + "...")

    st.subheader("🔒 Security & Compliance")
    # TODO Backend (SecurityAgent): Provide more granular requirements (data_retention, backup_recovery, HIPAA, PCI DSS etc.).
    st.write(
        f"**Compliance Suggested:** {', '.join(security.get('compliance', ['N/A (TODO)']))}"
    )
    st.json(
        security, expanded=False
    )  # Display whatever current security agent provides

    st.subheader("⚙️ Technology Stack Recommendations")
    # TODO Backend (TechAgent): Provide more detailed, context-aware stack (specific tools, mobile tech).
    st.json(tech, expanded=False)  # Display whatever current tech agent provides

    st.subheader("💰 Budget Analysis")
    # TODO Backend (EnhancedBudgetAgent): Provide `cost_breakdown` (dict with % impacts) for table & chart.
    # TODO Backend (EnhancedBudgetAgent): Provide `cost_per_user_first_year`.
    if budget:
        st.write(f"**Est. Development Cost:** ${budget.get('development_cost',0):,.2f}")
        st.write(f"**Est. Total First Year:** ${budget.get('total_first_year',0):,.2f}")
        st.write(
            f"**Market Adjustment Factor Applied:** {budget.get('market_adjustment_factor', 'N/A (from backend)')}"
        )
        if budget.get("cost_breakdown"):
            st.write("**Cost Breakdown (TODO: EnhancedBudgetAgent for data):**")
            st.json(budget["cost_breakdown"], expanded=False)  # Placeholder display
            # Placeholder for pie chart if data becomes available
            # factors = {k:v for k,v in budget['cost_breakdown'].items() if v != 0}
            # if factors:
            #     fig, ax = plt.subplots()
            #     ax.pie(factors.values(), labels=factors.keys(), autopct='%1.1f%%', startangle=90)
            #     st.pyplot(fig)
        else:
            st.caption(
                "Detailed cost breakdown for chart not available (TODO: Backend)."
            )
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
        st.metric("High Risk Proposals (TODO)", high_risk_an)

        st.subheader("📊 Vendor Comparison Charts (Placeholders - Needs Backend Data)")
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
    st.button("📊 Export Analytics Data (TODO)", disabled=True)


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
