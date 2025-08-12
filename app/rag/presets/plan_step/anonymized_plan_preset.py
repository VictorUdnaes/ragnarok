from enum import Enum
from rag.presets.abstract.abstract_step_preset import AbstractStepPreset
from model.llm_response_model import LLMResponse, ResponseType
from typing import Dict, Any
from pydantic import BaseModel, Field
from tools.planning_tool import PlanningTool
from model.plan_model import Plan
from model.anonymize_model import AnonymizedQuestion, DeanonymizedPlan
from prompts.prompt_manager import anonymizer_prompt, planner_prompt, deanonymize_prompt
from langchain_openai import ChatOpenAI
import logging
from enum import Enum
from rag.presets.preset_store import preset_store

logger = logging.getLogger("ApplicationService")

@preset_store.register("anonymizer_preset")
class AnonymizedPlanPreset(AbstractStepPreset):
    def __init__(self):
        super().__init__()

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

    # RUNNABLE SEQUENCES ------------------------------------------------------------------------------------------------
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
        __class__.__name__
        return self.execute_runnable_sequence(
            method_name="deanonymize_plan",
            input_variables={"plan": plan, "mapping": mapping},
            prompt=deanonymize_prompt,
            format_object=Plan
        )