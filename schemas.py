from pydantic import BaseModel
from typing import Literal

class ClauseDiff(BaseModel):
    change_type: Literal["No Material Change", "Wording Modified", "Obligation Shifted"]
    summary: str
    risk: Literal["Low", "Medium", "High", "None"]