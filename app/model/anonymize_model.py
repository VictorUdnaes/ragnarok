from pydantic import BaseModel

class AnonymizedQuestion(BaseModel):
    anonymized_question: str
    mapping: dict
    explaination: str

class DeanonymizedPlan(BaseModel):
    plan: list