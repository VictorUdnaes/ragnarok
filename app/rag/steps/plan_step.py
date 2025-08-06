from enum import Enum
from rag.steps.abstract.abstract_rag_step import AbstractRagStep
from model.llm_response_model import LLMResponse, ResponseType
from typing import Dict, Any
from pydantic import BaseModel, Field
from tools.planning_tool import PlanningTool
from model.plan_model import Plan
from model.anonymize_model import AnonymizedQuestion, DeanonymizedPlan
from prompts.prompt_manager import anonymizer_prompt, planner_prompt, deanonymize_prompt
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger("ApplicationService")

# TODO: add functionality for not using anonymization
class PlanStep(AbstractRagStep):
    # TODO: standardize plan format to a list
    def execute(self) -> LLMResponse:
        anonymized_question_obj = self.anonymize_question(self.query)
        anonymized_plan = self.create_initial_plan(anonymized_question_obj.anonymized_question)

        plan: Plan = self.deanonymize_plan(
            plan=anonymized_plan.steps,
            mapping=anonymized_question_obj.mapping
        )

        logger.info(f"Plan created with {len(plan.steps)} steps:")
        for step in enumerate(plan.steps):
            logger.info(f"[bold green][{step}][/bold green]")

        self.response = LLMResponse(
            step_name="Anonymized Plan",
            response_type=ResponseType.LIST,
            data=plan.steps
        )

        return self.response

    def anonymize_question(self, question: str) -> AnonymizedQuestion:
        return self.execute_runnable_sequence(
            method_name="anonymize_question",
            input_variables={"question": question},
            prompt=anonymizer_prompt,
            format_object=AnonymizedQuestion
        )

    def create_initial_plan(self, question: str) -> Plan:
        return self.execute_runnable_sequence(
            method_name="create_initial_plan",
            input_variables={"question": question},
            prompt=planner_prompt,
            format_object=Plan
        )

    def deanonymize_plan(self, plan: str, mapping:str) -> Plan:
        return self.execute_runnable_sequence(
            method_name="deanonymize_plan",
            input_variables={"plan": plan, "mapping": mapping},
            prompt=deanonymize_prompt,
            format_object=Plan
        )
class PlanStepSpecification(BaseModel):
    query: str = Field(..., description="The original question to be planned.")
    plan: Plan = Field(..., description="The generated plan based on the question.")
    correction: Dict[str, str] = Field(default_factory=dict, description="Corrections to apply to the plan step.")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "plan": self.plan.model_dump(),
            "correction": self.correction
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlanStepSpecification':
        return cls(
            query=data.get("query", ""),
            plan=Plan(**data.get("plan", {})),
            correction=data.get("correction", {})
        )
