import ollama
import json
from schemas import ClauseDiff

def compare_clause(clause_id: str, a: str, b: str) -> dict:
    """Queries local Ollama instance with strict JSON schema enforcement."""
    prompt = f"""You are an elite legal contract auditor. Compare Version A and Version B of Clause [{clause_id}] for semantic mutations.
Ignore cosmetic or superficial formatting adjustments. If the fundamental meaning is unaltered, choose 'No Material Change'.

VERSION A:
{a}

VERSION B:
{b}
"""

    response = ollama.chat(
        model="qwen2.5:7b",  # Ensure you ran `ollama pull qwen2.5:7b` (or swap with phi4-mini)
        messages=[
            {"role": "system", "content": "You analyze legal text variances. You must return data matching the requested JSON schema configuration exactly."},
            {"role": "user", "content": prompt}
        ],
        format=ClauseDiff.model_json_schema(),
        options={"temperature": 0.0}
    )

    result_content = response["message"]["content"]
    
    # Defensive parsing pipeline
    try:
        return json.loads(result_content)
    except (json.JSONDecodeError, TypeError):
        try:
            # Fallback: force Pydantic to repair minor formatting quirks
            validated = ClauseDiff.model_validate_json(result_content)
            return validated.model_dump()
        except Exception:
            return {
                "change_type": "Wording Modified",
                "summary": "Mismatched edits found. Unable to parse structured JSON from local model.",
                "risk": "None"
            }