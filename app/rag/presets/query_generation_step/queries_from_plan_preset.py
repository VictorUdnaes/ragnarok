from rag.presets.abstract.abstract_step_preset import AbstractStepPreset
from prompts.prompt_manager import queries_from_plan_prompt
from model.llm_response_model import LLMResponse, ResponseType
from model.queries_from_plan import QueriesFromPlan
from logging import getLogger
from rag.presets.preset_store import preset_store
from pydantic import BaseModel

logger = getLogger("QueryGenerationPreset")

class QueriesFromPlanSpec(BaseModel):
    question: str
    plan_steps: list[str]

@preset_store.register("queries_from_plan_preset")
class QueriesFromPlanPreset(AbstractStepPreset):
    """Encapsulates logic for generating search queries from a plan.

    Usage contract (like other presets):
    - Set self.query (user question) and self.plan_obj or self.plan_steps before execute().
    - execute() will populate self.response (LLMResponse with list of queries).
    - Supports correction via register_correction/should_rerun from AbstractStepPreset.
    """

    def __init__(self):
        super().__init__()
        self.spec_class = QueriesFromPlanSpec

    def execute(self, spec: QueriesFromPlanSpec) -> LLMResponse:
        logger.info("Starting query generation preset execution")

        generated = self.generate_queries_from_plan(
            question=spec.question,
            plan_steps=spec.plan_steps,
        ).queries

        logger.info(f"Generated {len(generated)} queries from plan")
        for q in generated:
            logger.info(f"[bold green][{q}][/bold green]")

        self.response = LLMResponse(
            step_name="Query Generation",
            response_type=ResponseType.LIST,
            data=generated,
        )
        return self.response

    # Runnable sequence wrapper
    def generate_queries_from_plan(self, question: str, plan_steps: list[str]) -> QueriesFromPlan:
        logger.info("Creating queries from plan")
        return self.execute_runnable_sequence(
            method_name="create_queries_from_plan",
            input_variables={"question": question, "plan": plan_steps},
            prompt=queries_from_plan_prompt,
            format_object=QueriesFromPlan,
        )
