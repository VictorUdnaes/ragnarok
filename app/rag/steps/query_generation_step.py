from rag.steps.abstract.abstract_rag_step import AbstractRagStep
from prompts.prompt_manager import queries_from_plan_prompt
from model.llm_response_model import LLMResponse, ResponseType
from model.plan_model import Plan
from model.queries_from_plan import QueriesFromPlan
from logging import getLogger

logger = getLogger("QueryGenerationStep")

class QueryGenerationStep(AbstractRagStep):
    def __init__(self, query: str, plan_obj: Plan):
        self.query = query
        self.plan_obj = plan_obj
        self.correctionDict: dict[str, str] = {}

    def execute(self) -> LLMResponse:
        self.generated_queries = self.create_queries_from_plan(
                question=self.query,
                plan=self.plan_obj
        ).queries

        logger.info(f"Generated {len(self.generated_queries)} queries from plan:")
        for query in self.generated_queries:
            logger.info(f"[bold green][{query}][/bold green]")

        self.response = LLMResponse(
            step_name="Query Generation",
            response_type=ResponseType.LIST,
            data=self.generated_queries
        )
        
        self.correction = {}
        return self.response

    def retry(self) -> LLMResponse:
        logger.info("Retrying query generation step execution")
        return self.execute()

    def rerun_with_correction(self, correction: dict[str, str]) -> LLMResponse:
        self.correctionDict = correction if correction else {}
        logger.info("Rerunning query generation step with correction")
        return self.execute()

    def create_queries_from_plan(self, question: str, plan: Plan) -> QueriesFromPlan:
        logger.info("Creating queries from plan")
        runnable = None
        if self.correctionDict.get("create_queries_from_plan"):
            runnable = self.build_correction_sequence(
                original_input_variables={"question": question, "plan": plan.steps},
                original_prompt=queries_from_plan_prompt,
                format_object=QueriesFromPlan,
                correction=self.correctionDict["create_queries_from_plan"]
            )
        else:
            runnable = self.build_runnable_sequence(["question", "plan"], queries_from_plan_prompt, QueriesFromPlan)

        return runnable.invoke({
            "question": question,
            "plan": plan.steps
        })