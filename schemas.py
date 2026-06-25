from pydantic import BaseModel
from typing import Literal

class ClauseDiff(BaseModel):
    is_same_topic: bool  # LLM acts as the matching engine
    change_type: Literal["Wording Modified", "Obligation Shifted", "Completely Different"]
    summary: str
    risk: Literal["Low", "Medium", "High", "None"]