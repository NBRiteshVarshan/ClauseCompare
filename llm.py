import ollama
import json
from schemas import ClauseDiff

def compare_clause(a: str, b: str) -> dict:
    """Queries local Ollama to determine if two text blocks share the same legal topic."""
    prompt = f"""You are a semantic analyzer. Your job is to determine if Text A and Text B are discussing the SAME underlying topic, rule, or obligation, even if the wording or language is completely different.

CRITICAL INSTRUCTIONS:
1. If they discuss completely different topics (e.g., one is about Termination, the other is about Payment), set "is_same_topic" to false and "change_type" to "Completely Different".
2. If they DO discuss the same topic, set "is_same_topic" to true. Then analyze how the obligation changed.
3. Pay aggressive attention to numbers, dates, and money. If those change, it is an "Obligation Shifted".

TEXT A:
{a}

TEXT B:
{b}
"""

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[
            {"role": "system", "content": "You analyze text variations. You must return data conforming strictly to the requested JSON schema."},
            {"role": "user", "content": prompt}
        ],
        format=ClauseDiff.model_json_schema(),
        options={"temperature": 0.0}
    )

    result_content = response["message"]["content"]
    
    try:
        return json.loads(result_content)
    except (json.JSONDecodeError, TypeError):
        try:
            validated = ClauseDiff.model_validate_json(result_content)
            return validated.model_dump()
        except Exception:
            return {
                "is_same_topic": False,
                "change_type": "Completely Different",
                "summary": "JSON parsing error.",
                "risk": "None"
            }