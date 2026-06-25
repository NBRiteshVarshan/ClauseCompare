import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time
import numpy as np

from document_processor import ClauseExtractor, get_document_summary
from clause_matcher import LegalClauseMatcher
from utils import format_report, save_report

# Page configuration
st.set_page_config(
    page_title="Legal Document Comparator",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .clause-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3B82F6;
    }
    .diff-box {
        background-color: #FEF2F2;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #EF4444;
    }
    .match-box {
        background-color: #F0FDF4;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #22C55E;
    }
    .exact-box {
        background-color: #F0FDF4;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #059669;
    }
    .partial-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #D97706;
    }
    .unique-box {
        background-color: #FEF2F2;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #DC2626;
    }
    .stButton > button {
        width: 100%;
        background-color: #1E3A8A;
        color: white;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1E40AF;
        color: white;
    }
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = None
    if 'doc1_clauses' not in st.session_state:
        st.session_state.doc1_clauses = None
    if 'doc2_clauses' not in st.session_state:
        st.session_state.doc2_clauses = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'reports_generated' not in st.session_state:
        st.session_state.reports_generated = []
    if 'extraction_config' not in st.session_state:
        st.session_state.extraction_config = {
            'min_clause_length': 60,
            'merge_threshold': 30
        }

def categorize_matches(results, doc1_clauses, doc2_clauses):
    """
    Categorize clauses into three groups:
    1. Exact matches (similarity >= 0.999)
    2. Partial matches (0.5 <= similarity < 0.999)
    3. Unique clauses (similarity < 0.5) – from both documents
    """
    exact_matches = []      # list of (doc1_clause, doc2_clause, sim)
    partial_matches = []    # list of (doc1_clause, doc2_clause, sim)
    unique_clauses = []     # list of (clause_text, document_name)

    # Get matching details for doc1
    matching_details = results.get('matching_details', [])
    
    # For each doc1 clause, we have top_similarity and top_match_idx
    for detail in matching_details:
        sim = detail.get('top_similarity', 0.0)
        idx = detail.get('top_match_idx', -1)
        clause1_text = detail['clause_text']
        clause1_num = detail.get('clause_number', '')
        
        if idx >= 0 and idx < len(doc2_clauses):
            clause2 = doc2_clauses[idx]
            clause2_text = clause2['text']
            clause2_num = clause2.get('number', '')
            
            if sim >= 0.999:
                exact_matches.append({
                    'doc1_num': clause1_num,
                    'doc1_text': clause1_text,
                    'doc2_num': clause2_num,
                    'doc2_text': clause2_text,
                    'similarity': sim
                })
            elif sim >= 0.5:
                partial_matches.append({
                    'doc1_num': clause1_num,
                    'doc1_text': clause1_text,
                    'doc2_num': clause2_num,
                    'doc2_text': clause2_text,
                    'similarity': sim
                })
            else:
                # unique from doc1
                unique_clauses.append({
                    'text': clause1_text,
                    'document': 'Document 1',
                    'number': clause1_num,
                    'similarity': sim
                })
        else:
            # no match found (should not happen)
            unique_clauses.append({
                'text': clause1_text,
                'document': 'Document 1',
                'number': clause1_num,
                'similarity': 0.0
            })

    # Now for doc2 clauses that are not matched (similarity < 0.5) to any doc1
    # We'll use the doc2_best_similarities and doc2_best_indices from results if available
    # but we can also compute directly from the matched flags
    # We'll use the only_in_doc2 list (which already contains unmatched doc2 clauses)
    # but we need to include even those that might have similarity >= 0.5? Actually only_in_doc2 are those that didn't get matched via our algorithm (LLM or high-sim).
    # For the unique category, we want to include doc2 clauses that have best similarity < 0.5.
    # We can compute from doc2_best_similarities that we stored.
    doc2_best_sims = results.get('doc2_best_similarities', [])
    if doc2_best_sims:
        for j, sim in enumerate(doc2_best_sims):
            if sim < 0.5:
                clause = doc2_clauses[j]
                unique_clauses.append({
                    'text': clause['text'],
                    'document': 'Document 2',
                    'number': clause.get('number', str(j+1)),
                    'similarity': sim
                })
    else:
        # fallback: use only_in_doc2
        for clause in results.get('only_in_doc2', []):
            # check if its similarity is < 0.5 (we have it)
            sim = clause.get('similarity', 0.0)
            if sim < 0.5:
                unique_clauses.append({
                    'text': clause['text'],
                    'document': 'Document 2',
                    'number': clause.get('number', ''),
                    'similarity': sim
                })

    # Sort unique by similarity descending
    unique_clauses.sort(key=lambda x: x['similarity'], reverse=True)

    return exact_matches, partial_matches, unique_clauses

def main():
    st.markdown('<h1 class="main-header">⚖️ Legal Document Comparator</h1>', unsafe_allow_html=True)
    st.markdown("Compare two legal documents and identify differences in clauses")

    initialize_session_state()

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("📄 Extraction Settings")
        min_clause_length = st.slider(
            "Minimum Clause Length (characters)",
            min_value=20,
            max_value=200,
            value=60,
            step=10,
            help="Segments shorter than this will be filtered out as noise"
        )

        merge_threshold = st.slider(
            "Merge Short Segments (characters)",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="Segments shorter than this will be merged with the next segment"
        )

        st.session_state.extraction_config['min_clause_length'] = min_clause_length
        st.session_state.extraction_config['merge_threshold'] = merge_threshold

        st.subheader("🎯 Comparison Settings")
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.05,
            help="Lower values will find more potential matches but may include false positives"
        )

        st.subheader("🧠 LLM Settings")
        st.info("Using Qwen2.5:7b (Local)")
        st.caption("✅ No data leaves your machine")

        st.divider()

        if st.session_state.doc1_clauses and st.session_state.doc2_clauses:
            st.subheader("📊 Document Summary")
            summary1 = get_document_summary(st.session_state.doc1_clauses)
            summary2 = get_document_summary(st.session_state.doc2_clauses)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Document 1", summary1['total'], "clauses")
                st.caption(f"Avg length: {summary1['avg_length']} words")
                st.caption(f"Conditions: {summary1['has_conditions']}")
            with col2:
                st.metric("Document 2", summary2['total'], "clauses")
                st.caption(f"Avg length: {summary2['avg_length']} words")
                st.caption(f"Conditions: {summary2['has_conditions']}")

    # Main content - two columns for document upload
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Document 1")
        doc1_file = st.file_uploader(
            "Upload Document 1 (PDF or DOCX)",
            type=['pdf', 'docx'],
            key="doc1"
        )

        if doc1_file:
            if st.button("📥 Process Document 1", key="process1"):
                with st.spinner("Extracting clauses using text-block strategy..."):
                    extractor = ClauseExtractor(
                        min_clause_length=st.session_state.extraction_config['min_clause_length'],
                        merge_threshold=st.session_state.extraction_config['merge_threshold']
                    )
                    try:
                        file_bytes = doc1_file.read()
                        if doc1_file.type == 'application/pdf':
                            clauses = extractor.extract_from_pdf(file_bytes)
                        else:
                            clauses = extractor.extract_from_docx(file_bytes)

                        st.session_state.doc1_clauses = clauses
                        st.success(f"✅ Extracted {len(clauses)} clauses using text-block strategy")
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")

            if st.session_state.doc1_clauses:
                with st.expander(f"📋 View Clauses ({len(st.session_state.doc1_clauses)})"):
                    for idx, clause in enumerate(st.session_state.doc1_clauses[:5]):
                        st.markdown(f"""
                            <div class="clause-box">
                                <strong>Clause {clause.get('number', idx+1)}</strong>
                                <span style="color: #6B7280; font-size: 0.8rem;">
                                    ({clause['metadata']['word_count']} words)
                                    {'📌' if clause['metadata']['is_title'] else ''}
                                </span><br>
                                {clause['text'][:300]}...
                            </div>
                        """, unsafe_allow_html=True)
                    if len(st.session_state.doc1_clauses) > 5:
                        st.caption(f"... and {len(st.session_state.doc1_clauses) - 5} more clauses")

    with col2:
        st.subheader("📄 Document 2")
        doc2_file = st.file_uploader(
            "Upload Document 2 (PDF or DOCX)",
            type=['pdf', 'docx'],
            key="doc2"
        )

        if doc2_file:
            if st.button("📥 Process Document 2", key="process2"):
                with st.spinner("Extracting clauses using text-block strategy..."):
                    extractor = ClauseExtractor(
                        min_clause_length=st.session_state.extraction_config['min_clause_length'],
                        merge_threshold=st.session_state.extraction_config['merge_threshold']
                    )
                    try:
                        file_bytes = doc2_file.read()
                        if doc2_file.type == 'application/pdf':
                            clauses = extractor.extract_from_pdf(file_bytes)
                        else:
                            clauses = extractor.extract_from_docx(file_bytes)

                        st.session_state.doc2_clauses = clauses
                        st.success(f"✅ Extracted {len(clauses)} clauses using text-block strategy")
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")

            if st.session_state.doc2_clauses:
                with st.expander(f"📋 View Clauses ({len(st.session_state.doc2_clauses)})"):
                    for idx, clause in enumerate(st.session_state.doc2_clauses[:5]):
                        st.markdown(f"""
                            <div class="clause-box">
                                <strong>Clause {clause.get('number', idx+1)}</strong>
                                <span style="color: #6B7280; font-size: 0.8rem;">
                                    ({clause['metadata']['word_count']} words)
                                    {'📌' if clause['metadata']['is_title'] else ''}
                                </span><br>
                                {clause['text'][:300]}...
                            </div>
                        """, unsafe_allow_html=True)
                    if len(st.session_state.doc2_clauses) > 5:
                        st.caption(f"... and {len(st.session_state.doc2_clauses) - 5} more clauses")

    # Compare button (center aligned)
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        compare_btn = st.button(
            "🔄 Compare Documents",
            use_container_width=True,
            disabled=not (st.session_state.doc1_clauses and st.session_state.doc2_clauses)
        )

    # Run comparison
    if compare_btn and st.session_state.doc1_clauses and st.session_state.doc2_clauses:
        with st.spinner("Comparing documents... This may take a few minutes."):
            st.session_state.processing = True

            try:
                matcher = LegalClauseMatcher()
                results = matcher.match_documents(
                    st.session_state.doc1_clauses,
                    st.session_state.doc2_clauses,
                    doc1_name="Document 1",
                    doc2_name="Document 2",
                    similarity_threshold=similarity_threshold,
                    high_similarity_threshold=0.8
                )

                st.session_state.comparison_results = results
                st.session_state.processing = False

                report_text = format_report(results)
                report_file = save_report(results)
                st.session_state.reports_generated.append(report_file)

                st.success(f"✅ Comparison complete! Processed {results['total_doc1'] + results['total_doc2']} clauses in {results['processing_time']:.2f} seconds")

            except Exception as e:
                st.session_state.processing = False
                st.error(f"Error during comparison: {str(e)}")
                st.error("Make sure Ollama is running and qwen2.5:7b model is downloaded")
                st.info("To install and run Ollama:\n1. Download from https://ollama.ai\n2. Run: ollama pull qwen2.5:7b\n3. Run: ollama serve")

    # Display results
    if st.session_state.comparison_results and not st.session_state.processing:
        results = st.session_state.comparison_results
        doc1_clauses = st.session_state.doc1_clauses
        doc2_clauses = st.session_state.doc2_clauses

        # ---- Summary metrics ----
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="stats-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                    <h3>{results['total_doc1'] + results['total_doc2']}</h3>
                    <p>Total Clauses</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="stats-card" style="background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);">
                    <h3>{results['matching_count']}</h3>
                    <p>Matching Clauses</p>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="stats-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <h3>{len(results['only_in_doc1'])}</h3>
                    <p>Only in Doc 1</p>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div class="stats-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                    <h3>{len(results['only_in_doc2'])}</h3>
                    <p>Only in Doc 2</p>
                </div>
            """, unsafe_allow_html=True)

        # ---- Visualization ----
        st.subheader("📊 Visual Analysis")
        fig = go.Figure(data=[
            go.Bar(
                x=['Doc 1 Only', 'Doc 2 Only', 'Matching'],
                y=[len(results['only_in_doc1']), len(results['only_in_doc2']), results['matching_count']],
                marker_color=['#EF4444', '#3B82F6', '#22C55E'],
                text=[len(results['only_in_doc1']), len(results['only_in_doc2']), results['matching_count']],
                textposition='auto',
            )
        ])
        fig.update_layout(
            title='Clause Comparison Overview',
            xaxis_title='Category',
            yaxis_title='Number of Clauses',
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
        )
        fig.update_traces(textfont_size=16)
        st.plotly_chart(fig, use_container_width=True)

        # ---- Three Categories ----
        st.subheader("📂 Clause Categories")
        exact_matches, partial_matches, unique_clauses = categorize_matches(results, doc1_clauses, doc2_clauses)

        # Expandable sections for each category
        with st.expander(f"✅ Exact Matches (Similarity = 1.00) — {len(exact_matches)} pairs"):
            if exact_matches:
                for match in exact_matches:
                    st.markdown(f"""
                        <div class="exact-box">
                            <strong>📄 Document 1 – Clause {match['doc1_num']}</strong><br>
                            {match['doc1_text']}
                            <br><br>
                            <strong>📄 Document 2 – Clause {match['doc2_num']}</strong><br>
                            {match['doc2_text']}
                            <br><br>
                            <span style="color: #059669;">✅ Similarity: {match['similarity']:.3f}</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No exact matches found.")

        with st.expander(f"🟡 Partial Matches (0.5 ≤ Similarity < 1.00) — {len(partial_matches)} pairs"):
            if partial_matches:
                for match in partial_matches:
                    st.markdown(f"""
                        <div class="partial-box">
                            <strong>📄 Document 1 – Clause {match['doc1_num']}</strong><br>
                            {match['doc1_text']}
                            <br><br>
                            <strong>📄 Document 2 – Clause {match['doc2_num']}</strong><br>
                            {match['doc2_text']}
                            <br><br>
                            <span style="color: #D97706;">🔶 Similarity: {match['similarity']:.3f}</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No partial matches found.")

        with st.expander(f"🔴 Unique Clauses (Similarity < 0.5) — {len(unique_clauses)} clauses"):
            if unique_clauses:
                for clause in unique_clauses:
                    doc_name = clause['document']
                    st.markdown(f"""
                        <div class="unique-box">
                            <strong>📄 {doc_name} – Clause {clause['number']}</strong><br>
                            {clause['text']}
                            <br><br>
                            <span style="color: #DC2626;">❌ Best similarity: {clause['similarity']:.3f}</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("All clauses have a good match (similarity ≥ 0.5).")

        # ---- Download Report ----
        st.subheader("📋 Download Full Report")
        col1, col2 = st.columns(2)
        with col1:
            report_text = format_report(results)
            st.download_button(
                label="📥 Download Report (TXT)",
                data=report_text,
                file_name=f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        with col2:
            json_file = save_report(results)
            with open(json_file, 'r') as f:
                json_data = f.read()
            st.download_button(
                label="📥 Download Report (JSON)",
                data=json_data,
                file_name=f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        # ---- Debug Details ----
        with st.expander("🔧 Processing Details"):
            st.json({
                'extraction_method': 'Text-block strategy (\\n\\n split)',
                'min_clause_length': st.session_state.extraction_config['min_clause_length'],
                'merge_threshold': st.session_state.extraction_config['merge_threshold'],
                'similarity_threshold': similarity_threshold,
                'high_similarity_threshold': 0.8,
                'processing_time': f"{results['processing_time']:.2f} seconds",
                'total_clauses_processed': results['total_doc1'] + results['total_doc2'],
                'llm_matches': results.get('llm_matches', 0),
                'high_sim_matches': results.get('high_sim_matches', 0)
            })

if __name__ == "__main__":
    # Check if Ollama is running
    try:
        import ollama
        ollama.list()
    except Exception:
        st.warning("⚠️ Ollama not detected. Please ensure Ollama is running and qwen2.5:7b is downloaded.")
        st.info("""
        To install:
        1. Download Ollama from https://ollama.ai
        2. Run in terminal: `ollama pull qwen2.5:7b`
        3. Run in terminal: `ollama serve`
        """)

    main()