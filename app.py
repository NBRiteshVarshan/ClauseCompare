import streamlit as st
from agent import run_diff

# Set layout window orientation config
st.set_page_config(layout="wide")
st.title("📄 Zero-Leak Document Compliance Agent")
st.caption("Offline legal contract intelligence engine with selective local LLM reasoning capabilities.")

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload Baseline Document (A)", type=["pdf", "docx", "txt"])
with col2:
    file_b = st.file_uploader("Upload Modified Document (B)", type=["pdf", "docx", "txt"])

if st.button("Execute Semantic Analysis Run", type="primary"):
    if file_a and file_b:
        with st.spinner("Analyzing document modifications in parallel..."):
            result = run_diff(file_a, file_b)

        st.success("Analysis Complete")
        
        # 1. Added Sections Rendering 
        st.subheader("➕ Added Clauses (Present in B, Missing in A)")
        if result["added"]:
            for item in result["added"]:
                with st.expander(f"🟢 {item['clause']}"):
                    st.code(item['content'], language="text")
        else:
            st.info("No structural additions detected.")

        # 2. Removed Sections Rendering
        st.subheader("➖ Removed Clauses (Present in A, Missing in B)")
        if result["removed"]:
            for item in result["removed"]:
                with st.expander(f"🔴 {item['clause']}"):
                    st.code(item['content'], language="text")
        else:
            st.info("No structural deletions detected.")

        # 3. Semantic Analysis Summary Cards
        st.subheader("🔄 Modified Clause Semantic Evaluations")
        if result["modified"]:
            for clause, analysis in result["modified"].items():
                risk_indicator = "❌" if analysis['risk'] == "High" else "⚠️" if analysis['risk'] == "Medium" else "ℹ️"
                with st.expander(f"{risk_indicator} {clause} — {analysis['change_type']}"):
                    st.markdown(f"**Change Summary:** {analysis['summary']}")
                    st.markdown(f"**Risk Level:** `{analysis['risk']}`")
        else:
            st.info("No meaningful semantic variations found between matching headers.")
    else:
        st.error("Execution blocked: Please provide both source documents.")