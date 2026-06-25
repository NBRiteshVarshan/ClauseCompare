import re
import hashlib
import json
from typing import List, Dict, Any
from datetime import datetime
import numpy as np

# ------------------------------------------------------------
# Helper to recursively convert NumPy types to Python natives
# ------------------------------------------------------------
def sanitize_for_json(obj):
    """
    Recursively convert any NumPy types to native Python types
    so that json.dump works without custom encoders.
    """
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    return obj

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def clean_text(text: str) -> str:
    """Clean and normalize text for processing"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\;\:\-\"\'\?\$\%\&\(\)]', '', text)
    return text.strip()

def generate_clause_id(text: str, document_name: str) -> str:
    """Generate unique ID for each clause for caching"""
    content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    doc_hash = hashlib.md5(document_name.encode()).hexdigest()[:6]
    return f"{doc_hash}_{content_hash}"

def format_report(results: Dict) -> str:
    """Format comparison results as a readable report"""
    report = []
    report.append("=" * 80)
    report.append("LEGAL DOCUMENT COMPARISON REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)

    # Clauses only in Document 1
    report.append("\n📄 CLAUSES IN DOCUMENT 1 BUT NOT IN DOCUMENT 2")
    report.append("-" * 60)
    for idx, clause in enumerate(results.get('only_in_doc1', []), 1):
        report.append(f"{idx}. {clause['text']}")
        if clause.get('closest_match'):
            report.append(f"   → Closest match in Doc2: {clause['closest_match'][:100]}...")
            report.append(f"   → Similarity score: {clause.get('similarity', 0):.2f}")
        report.append("")

    # Clauses only in Document 2
    report.append("\n📄 CLAUSES IN DOCUMENT 2 BUT NOT IN DOCUMENT 1")
    report.append("-" * 60)
    for idx, clause in enumerate(results.get('only_in_doc2', []), 1):
        report.append(f"{idx}. {clause['text']}")
        if clause.get('closest_match'):
            report.append(f"   → Closest match in Doc1: {clause['closest_match'][:100]}...")
            report.append(f"   → Similarity score: {clause.get('similarity', 0):.2f}")
        report.append("")

    # Summary statistics
    report.append("\n📊 SUMMARY STATISTICS")
    report.append("-" * 60)
    report.append(f"Total clauses in Document 1: {results.get('total_doc1', 0)}")
    report.append(f"Total clauses in Document 2: {results.get('total_doc2', 0)}")
    report.append(f"Clauses only in Document 1: {len(results.get('only_in_doc1', []))}")
    report.append(f"Clauses only in Document 2: {len(results.get('only_in_doc2', []))}")
    report.append(f"Matching clauses found: {results.get('matching_count', 0)}")
    report.append(f"Comparison time: {results.get('processing_time', 0):.2f} seconds")
    report.append("=" * 80)

    return "\n".join(report)

def save_report(results: Dict, filename: str = None):
    """
    Save comparison results to a JSON file.
    This version explicitly sanitises all NumPy types before serialisation.
    """
    if not filename:
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # 1. Build a dictionary with only the keys we want
    clean_results = {
        'only_in_doc1': results.get('only_in_doc1', []),
        'only_in_doc2': results.get('only_in_doc2', []),
        'total_doc1': results.get('total_doc1', 0),
        'total_doc2': results.get('total_doc2', 0),
        'matching_count': results.get('matching_count', 0),
        'processing_time': results.get('processing_time', 0),
        'timestamp': datetime.now().isoformat()
    }

    # 2. Sanitise the entire dictionary (recursive)
    sanitised = sanitize_for_json(clean_results)

    # 3. Write JSON – now all values are standard Python types
    with open(filename, 'w') as f:
        json.dump(sanitised, f, indent=2)

    return filename

# ------------------------------------------------------------
# (Optional) Debug helper – find any stray float32 in results
# ------------------------------------------------------------
def find_numpy_types(obj, path=""):
    """Print locations of any NumPy types in a nested structure."""
    if isinstance(obj, (np.float32, np.float64, np.int32, np.int64, np.ndarray)):
        print(f"Found {type(obj)} at: {path}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            find_numpy_types(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            find_numpy_types(item, f"{path}[{i}]")