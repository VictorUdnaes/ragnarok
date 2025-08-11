from rag.steps.abstract.abstract_step_preset import AbstractStepPreset
from model.llm_response_model import LLMResponse, ResponseType
from prompts.prompt_manager import analysis_prompt
from model.response_model import RAGResponse
from langchain_core.documents import Document
from logging import getLogger
from model.plan_model import Plan

logger = getLogger("FinalResponsePreset")


class DefaultFinalResponsePreset(AbstractStepPreset):
    def __init__(self):
        super().__init__()
        self.documents: list[Document] = []

    def execute(self) -> LLMResponse:
        logger.info("Starting final response preset execution")
        generated_response = self.generate_final_response()
        self.response = LLMResponse(
            step_name="Final Response",
            response_type=ResponseType.TEXT,
            data=generated_response,
        )
        return self.response

    def generate_final_response(self, documents: list[Document], plan_obj: Plan, query: str, generated_queries: list[str]) -> RAGResponse:
        logger.info("Generating final response from documents and plan")
        response = self.execute_runnable_sequence(
            method_name="generate_final_response",
            input_variables={
                "documents": documents,
                "plan": plan_obj,
                "query": query,
                "generated_queries": generated_queries,
            },
            prompt=analysis_prompt,
            format_object=RAGResponse,
        )

        logger.info(f"[bold blue]Final answer: {response}[/bold blue]")
        logger.info("[bold green]RAG chain executed successfully.[/bold green]")
        return response
