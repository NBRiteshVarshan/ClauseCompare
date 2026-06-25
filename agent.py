from parser import extract_text, split_clauses
from llm import compare_clause

def run_diff(file_a, file_b) -> dict:
    """Executes a dual-pass cross-cartesian semantic match between two documents."""
    bytes_a = file_a.getvalue()
    bytes_b = file_b.getvalue()
    
    text_a = extract_text(file_a.name, bytes_a)
    text_b = extract_text(file_b.name, bytes_b)

    doc_a = split_clauses(text_a)
    doc_b = split_clauses(text_b)

    report = {
        "added": [],
        "removed": [],
        "modified": {}
    }

    matched_b_keys = set()
    matched_a_keys = set()

    # ==========================================
    # PASS 1: Exact Match Fast-Track (Saves CPU)
    # ==========================================
    for key_a, val_a in doc_a.items():
        for key_b, val_b in doc_b.items():
            if key_b in matched_b_keys:
                continue
            # If the text is perfectly identical, pair them instantly
            if val_a.strip() == val_b.strip():
                matched_a_keys.add(key_a)
                matched_b_keys.add(key_b)
                break

    # Extract only the clauses that actually have differences
    remaining_a = {k: v for k, v in doc_a.items() if k not in matched_a_keys}
    remaining_b = {k: v for k, v in doc_b.items() if k not in matched_b_keys}

    # ==========================================
    # PASS 2: LLM Semantic Cross-Matching
    # ==========================================
    for key_a, text_a in remaining_a.items():
        match_found = False
        
        # We iterate over a list so we can safely delete items from remaining_b when found
        for key_b, text_b in list(remaining_b.items()):
            print(f"[DEBUG] Comparing {key_a} against {key_b}...")
            
            # Call the LLM to ask: "Are these the same topic?"
            analysis = compare_clause(text_a, text_b)
            
            if analysis.get("is_same_topic"):
                print(f" -> MATCH FOUND! {key_a} maps to {key_b}")
                match_found = True
                matched_b_keys.add(key_b)
                del remaining_b[key_b] # Mark Document 2 clause as True (remove from pool)
                
                # Render the UI name correctly if headers changed
                ui_identifier = f"{key_a} ➔ {key_b}" if key_a != key_b else key_a
                
                # Sanitize risk capitalization
                raw_risk = analysis.get("risk", "Low")
                analysis["risk"] = raw_risk.strip().capitalize() if raw_risk else "None"
                
                report["modified"][ui_identifier] = analysis
                break # Break inner loop, move to next Document 1 clause
                
        # If it checked every clause in Doc 2 and found no match, it was removed
        if not match_found:
            report["removed"].append({"clause": key_a, "content": text_a})

    # ==========================================
    # PASS 3: Catch Leftovers in Document 2
    # ==========================================
    # Any clause in Doc 2 not marked as matched is a brand new addition
    for key_b, text_b in remaining_b.items():
        report["added"].append({"clause": key_b, "content": text_b})

    return report