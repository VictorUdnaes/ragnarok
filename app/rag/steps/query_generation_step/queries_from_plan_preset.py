from rag.steps.abstract.abstract_step_preset import AbstractStepPreset
from prompts.prompt_manager import queries_from_plan_prompt
from model.llm_response_model import LLMResponse, ResponseType
from model.queries_from_plan import QueriesFromPlan
from logging import getLogger

logger = getLogger("QueryGenerationPreset")


class QueriesFromPlanPreset(AbstractStepPreset):
    """Encapsulates logic for generating search queries from a plan.

    Usage contract (like other presets):
    - Set self.query (user question) and self.plan_obj or self.plan_steps before execute().
    - execute() will populate self.response (LLMResponse with list of queries).
    - Supports correction via register_correction/should_rerun from AbstractStepPreset.
    """

    def __init__(self):
        super().__init__()
        self.plan_steps: list[str] = []

    def execute(self) -> LLMResponse:
        logger.info("Starting query generation preset execution")

        # Prefer explicitly set plan_steps, otherwise try to read from plan_obj
        plan_steps = self.plan_steps or (self.plan_obj.steps if getattr(self, "plan_obj", None) else [])

        generated = self.generate_queries_from_plan(
            question=self.query,
            plan_steps=plan_steps,
        ).queries

        logger.info(f"Generated {len(generated)} queries from plan")
        for q in generated:
            logger.info(f"[bold green][{q}][/bold green]")

        self.generated_queries = generated

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
