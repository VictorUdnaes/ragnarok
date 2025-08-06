from rag.steps.abstract.abstract_rag_step import AbstractRagStep
from model.llm_response_model import LLMResponse, ResponseType
from typing import Dict, Any
from pydantic import BaseModel, Field
from tools.planning_tool import PlanningTool
from model.plan_model import Plan
from model.anonymize_model import AnonymizedQuestion, DeanonymizedPlan
from prompts.prompt_manager import anonymizer_prompt, planner_prompt, deanonymize_prompt
from langchain_openai import ChatOpenAI
from logging import getLogger

logger = getLogger("PlanStep")

# TODO: add functionality for not using anonymization
class PlanStep(AbstractRagStep):
    def __init__(self, llm: ChatOpenAI, query: str):
        self.llm = llm
        self.query = query
        self.planning_tool = PlanningTool(llm=llm)
        self.plan: Plan = None
        self.correctionDict: dict[str, str] = {}

    # TODO: standardize plan format to a list
    def execute(self) -> LLMResponse:
        anonymized_question_obj = self.anonymize_question(self.query)
        anonymized_plan = self.create_initial_plan(anonymized_question_obj.anonymized_question)

        self.plan: Plan = self.deanonymize_plan(
            plan=anonymized_plan.steps,
            mapping=anonymized_question_obj.mapping
        )

        logger.info(f"Plan created with {len(self.plan.steps)} steps:")
        for step in enumerate(self.plan.steps):
            logger.info(f"[bold green][{step}][/bold green]")

        self.response = LLMResponse(
            step_name="Anonymized Plan",
            response_type=ResponseType.LIST,
            data=self.plan.steps
        )

        # Reset correction after plan is generated
        self.correction = {}
        return self.response

    def retry(self) -> LLMResponse:
        logger.info("Retrying plan step execution")
        return self.execute()

    def rerun_with_correction(self, response: LLMResponse, correction: dict[str, str]) -> LLMResponse:
        self.correctionDict = correction if correction else {}
        logger.info("Rerunning plan step with correction")
        return self.execute()

    # Util methods for generating the plan 
    def anonymize_question(self, question: str) -> AnonymizedQuestion:
        runnable = None
        if self.correctionDict.get("anonymize_question"):
            runnable = self.build_correction_sequence(
                original_input_variables={"question": question},
                original_prompt=anonymizer_prompt,
                format_object=AnonymizedQuestion,
                correction=self.correctionDict["anonymize_question"]
            )
        else:
            runnable = self.build_runnable_sequence(["question"], planner_prompt, Plan)
        
        return runnable.invoke(input=question)

    def create_initial_plan(self, question: str) -> Plan:
        if self.correctionDict.get("create_initial_plan"):
            runnable = self.build_correction_sequence(
                original_input_variables={"question": question},
                original_prompt=planner_prompt,
                format_object=Plan,
                correction=self.correctionDict["create_initial_plan"]
            )
        else:
            runnable = self.build_runnable_sequence(["question"], planner_prompt, Plan)
        
        return runnable.invoke(input=question)


    def deanonymize_plan(self, plan: str, mapping:str) -> Plan:
        if self.correctionDict.get("deanonymize_plan"):
            runnable = self.build_correction_sequence(
                original_input_variables={"plan": plan, "mapping": mapping},
                original_prompt=deanonymize_prompt,
                format_object=Plan,
                correction=self.correctionDict["deanonymize_plan"]
            )
        else:
            runnable = self.build_runnable_sequence(["plan", "mapping"], deanonymize_prompt, Plan)
        
        return runnable.invoke({
            "plan": plan,
            "mapping": mapping
        })
    
class PlanStepSpecification(BaseModel):
    query: str = Field(..., description="The original question to be planned.")
    plan: Plan = Field(..., description="The generated plan based on the question.")
    correction: Dict[str, str] = Field(default_factory=dict, description="Corrections to apply to the plan step.")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "plan": self.plan.dict(),
            "correction": self.correction
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlanStepSpecification':
        return cls(
            query=data.get("query", ""),
            plan=Plan(**data.get("plan", {})),
            correction=data.get("correction", {})
        )

