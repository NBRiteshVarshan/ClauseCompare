import streamlit as st
from agent import run_diff

st.set_page_config(layout="wide")
st.title("📄 Zero-Leak Document Compliance Agent")
st.caption("Offline legal contract intelligence engine with Cross-Cartesian Semantic Mapping.")

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload Baseline Document (A)", type=["pdf", "docx", "txt"])
with col2:
    file_b = st.file_uploader("Upload Modified Document (B)", type=["pdf", "docx", "txt"])

if st.button("Execute Semantic Analysis Run", type="primary"):
    if file_a and file_b:
        with st.spinner("Executing semantic multi-pass analysis. This may take a moment on local hardware..."):
            result = run_diff(file_a, file_b)

        st.success("Analysis Complete")
        
        # 1. Added Sections (Found in B, not in A)
        st.subheader("➕ Added Clauses")
        if result["added"]:
            for item in result["added"]:
                with st.expander(f"🟢 {item['clause']}"):
                    st.code(item['content'], language="text")
        else:
            st.info("No structural additions detected.")

        # 2. Removed Sections (Found in A, not in B)
        st.subheader("➖ Removed Clauses")
        if result["removed"]:
            for item in result["removed"]:
                with st.expander(f"🔴 {item['clause']}"):
                    st.code(item['content'], language="text")
        else:
            st.info("No structural deletions detected.")

        # 3. Modified Clauses
        st.subheader("🔄 Modified Clause Semantic Evaluations")
        if result["modified"]:
            for clause, analysis in result["modified"].items():
                risk_indicator = "❌" if analysis['risk'] == "High" else "⚠️" if analysis['risk'] == "Medium" else "ℹ️"
                with st.expander(f"{risk_indicator} {clause} — {analysis['change_type']}"):
                    st.markdown(f"**Change Summary:** {analysis['summary']}")
                    st.markdown(f"**Risk Level:** `{analysis['risk']}`")
        else:
            st.info("No semantic variations found.")
    else:
        st.error("Execution blocked: Please provide both source documents.")