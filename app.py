import streamlit as st

from rfp_automation.workflow.enhanced_workflow import EnhancedRFPAutomationWorkflow
from rfp_automation.config.settings import get_settings

settings = get_settings()


def enhanced_streamlit_ui():
    """Enhanced Streamlit UI with search and vector storage insights"""
    st.title("🚀 AI-Powered RFP Automation with Market Intelligence")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        tavily_api_key = settings.TAVILY_API_KEY
        use_market_research = st.checkbox("Enable Market Research", value=True)
        use_knowledge_cache = st.checkbox("Use Knowledge Cache", value=True)

    # Main workflow
    user_input = st.text_area(
        "Describe your project requirements:",
        placeholder="We need a mobile app for logistics tracking, scalable to 100k users",
    )

    if st.button("Generate RFP with Market Intelligence"):

        with st.spinner("Conducting market research and generating RFP..."):
            workflow = EnhancedRFPAutomationWorkflow(tavily_api_key)
            result = workflow.run(user_input)

        # Display results with enhanced insights
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Market Intelligence")
            if result.get("market_research"):
                market_data = result["market_research"]
                st.write("**Market Trends:**")
                for trend in market_data.get("market_trends", []):
                    st.write(f"• {trend}")

                st.write("**Search Metadata:**")
                search_meta = result.get("search_metadata", {})
                st.metric("Sources Found", search_meta.get("sources_found", 0))
                st.metric("Searches Performed", search_meta.get("total_searches", 0))

        with col2:
            st.subheader("🧠 Knowledge Base")
            cached_knowledge = result.get("cached_knowledge", [])
            if cached_knowledge:
                st.write(f"Found {len(cached_knowledge)} relevant cached insights")
                for knowledge in cached_knowledge[:3]:
                    st.write(f"• {knowledge.get('content', '')[:100]}...")
            else:
                st.write("No cached knowledge found - building knowledge base...")

        # Enhanced RFP display
        st.subheader("📄 Generated RFP")
        st.markdown(result["rfp_document"])

        # Enhanced vendor analysis
        st.subheader("🏢 Intelligent Vendor Analysis")
        vendor_intel = result.get("vendor_intelligence", {})
        if vendor_intel.get("research_conducted"):
            st.success(
                f"✅ Researched {vendor_intel.get('vendors_researched', 0)} vendors using market intelligence"
            )

        # Display proposals with market context
        for i, proposal in enumerate(result["vendor_proposals"]):
            with st.expander(f"Proposal {i+1}: {proposal['vendor_name']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Cost", f"${proposal['cost']:,.0f}")
                with col2:
                    st.metric("Timeline", f"{proposal['timeline_months']} months")
                with col3:
                    market_based = (
                        "✅" if proposal.get("market_intelligence_used") else "❌"
                    )
                    st.metric("Market Intelligence", market_based)

                st.write("**Features:**", ", ".join(proposal["features"]))
                if proposal.get("strengths"):
                    st.write("**Strengths:**", ", ".join(proposal["strengths"]))


if __name__ == "__main__":
    enhanced_streamlit_ui()
