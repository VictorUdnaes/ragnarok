from pydantic import BaseModel

class AnonymizedQuestion(BaseModel):
    question: str
    mapping: dict
    explaination: str

class DeanonymizedPlan(BaseModel):
    plan: list