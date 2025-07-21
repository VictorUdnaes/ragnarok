from pydantic import BaseModel, Field
from typing import List

class Plan(BaseModel):
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )