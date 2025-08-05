from rag.steps.abstract.abstract_rag_step import AbstractRagStep
from model.llm_response_model import LLMResponse, ResponseType
from services.vector_store import VectorStore
from prompts.prompt_manager import analysis_prompt
from model.response_model import RAGResponse
from langchain_core.documents import Document
from langchain_core.runnables import RunnableSequence
from typing import Type
from pydantic import BaseModel
from model.plan_model import Plan
from logging import getLogger

logger = getLogger("FinalResponseStep")


class FinalResponseStep(AbstractRagStep):
    def __init__(self, documents: list[Document], plan_obj: Plan, query: str, generated_queries: list[str]):
        self.documents = documents
        self.plan_obj = plan_obj
        self.query = query
        self.generated_queries = generated_queries
        self.correctionDict: dict[str, str] = {}

    def execute(self):
        logger.info("Starting final response step")
        self.response = self.generate_final_response()
        self.correctionDict = {}
        
        return LLMResponse(
            step_name="Final Response",
            response_type=ResponseType.TEXT,
            data=self.response
        )

    def retry(self) -> LLMResponse:
        logger.info("Retrying final response step execution")
        return self.execute()
    
    def rerun_with_correction(self, correction: dict[str, str]) -> LLMResponse:
        self.correctionDict = correction if correction else {}
        logger.info("Rerunning final response step with correction")
        return self.execute()
    
    def generate_final_response(self) -> RAGResponse:
        runnable: RunnableSequence = None
        if self.correctionDict.get("generate_final_response"):
            runnable = self.build_correction_sequence(
                original_input_variables={
                    "context": self.documents,
                    "plan": self.plan_obj.steps,
                    "original_question": self.query,
                    "generated_queries_from_plan": self.generated_queries
                },
                original_prompt=analysis_prompt,
                format_object=RAGResponse,
                correction=self.correctionDict["generate_final_response"]
            )
        else:
            runnable = self.build_runnable_sequence(
                input_variables=["context", "plan", "original_question", "generated_queries_from_plan"],
                prompt=analysis_prompt,
                format_object=RAGResponse
            )

        logger.info("Executing final response generation")
        # Invoke the runnable with the necessary input variables
        answer: RAGResponse = runnable.invoke({
            "context": self.documents,
            "plan": self.plan_obj.steps,
            "original_question": self.query,
            "generated_queries_from_plan": self.generated_queries
        })

        logger.info(f"[bold blue]Final answer: {answer}[/bold blue]")
        logger.info("[bold green]RAG chain executed successfully.[/bold green]")

        return answer
