from langchain_ollama import ChatOllama
from rag.presets.abstract.abstract_step_preset import AbstractStepPreset
from model.llm_response_model import LLMResponse, ResponseType
from prompts.prompt_manager import analysis_prompt
from model.response_model import RAGResponse
from langchain_core.documents import Document
from logging import getLogger
from model.plan_model import Plan
from rag.presets.preset_store import preset_store
    
logger = getLogger("FinalResponsePreset")

class DefaultFinalResponseSpec():
    def __init__(self):
        self.documents: list[Document] = []
        self.plan_steps: list[str] = []
        self.query: str = ""
        self.generated_queries: list[str] = []

@preset_store.register("default_final_response_preset")
class DefaultFinalResponsePreset(AbstractStepPreset):
    def __init__(self):
        super().__init__()
        self.documents: list[Document] = []
        self.spec_class = DefaultFinalResponseSpec

    def execute(
            self, 
            spec: DefaultFinalResponseSpec
        ) -> LLMResponse:
        logger.info("Starting final response preset execution")
        generated_response = self.generate_final_response(
            documents=spec.documents,
            plan_steps=spec.plan_steps,
            query=spec.query,
            generated_queries=spec.generated_queries
        )

        self.response = LLMResponse(
            step_name="Final Response",
            response_type=ResponseType.TEXT,
            data=generated_response,
        )
        return self.response

    def generate_final_response(self, documents: list[Document], plan_steps: list[str], query: str, generated_queries: list[str]) -> RAGResponse:
        logger.info("Generating final response from documents and plan")
        response = self.execute_runnable_sequence(
            method_name="generate_final_response",
            input_variables={
                "context": documents,
                "generated_queries_from_plan": generated_queries,
                "original_question": query,
                "plan": plan_steps,
            },
            prompt=analysis_prompt,
            format_object=RAGResponse,
        )

        logger.info(f"[bold blue]Final answer: {response}[/bold blue]")
        logger.info("[bold green]RAG chain executed successfully.[/bold green]")
        return response
