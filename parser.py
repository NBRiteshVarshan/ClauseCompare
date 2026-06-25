import fitz 
import docx2txt
import re
import tempfile
import os
import streamlit as st

@st.cache_data(show_spinner=False)
def extract_text(file_name: str, file_bytes: bytes) -> str:
    """Safely extracts text from document bytes with memory-safe caching."""
    if file_name.endswith(".pdf"):
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            # High-speed list comprehension prevents string memory thrashing
            pages = [page.get_text("text") for page in doc]
        return "\n".join(pages)

    elif file_name.endswith(".docx"):
        # docx2txt requires a physical file path, so we use a safe temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        
        try:
            text = docx2txt.process(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return text

    else:
        return file_bytes.decode("utf-8")

@st.cache_data(show_spinner=False)
def split_clauses(text: str) -> dict[str, str]:
    """Slices raw contract text into structured dictionary keys based on numbering."""
    # Splitting logic looks for headers like Section 1.2, Clause 3, Article 4.1
    pattern = r'(?=\b(?:Section|Clause|Article)\s+\d+(?:\.\d+)*)'
    parts = re.split(pattern, text)

    clauses = {}
    for p in parts:
        p = p.strip()
        if not p:
            continue

        match = re.match(r'^(Section|Clause|Article)\s+\d+(?:\.\d+)*', p)
        key = match.group(0) if match else "Preamble"

        if key in clauses:
            clauses[key] += "\n" + p
        else:
            clauses[key] = p

    return clauses