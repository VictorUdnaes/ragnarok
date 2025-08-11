import logging
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from rag.steps.retrieval_step.document_retrieval_step import DocumentRetrievalStep
from rag.steps.final_response_step.final_response_step import FinalResponseStep
from rag.steps.query_generation_step.query_generation_step import QueryGenerationStep
from services.vector_store import VectorStore
from tools.planning_tool import PlanningTool
from model.plan_model import Plan
from config.openai_config import openapi_client
from rag.steps.plan_step.plan_step import PlanStep

logger = logging.getLogger("ApplicationService")

class Pipeline():
    def __init__(
            self, 
            vectorstore: VectorStore, 
            embeddings: OllamaEmbeddings,
            llm: ChatOllama, 
            query: str
        ):
        self.vectorstore = vectorstore
        self.query = query
        self.llm = llm
        self.plan_obj: Plan | None = None
        self.generated_queries: list[str] = []
        self.available_retrievers: list[str] = ["chunk", "quote"]
        self.steps: dict[str, object] = {}
        self._steps_list: list[object] = []
        # RAG Steps
        self.plan_step = PlanStep()
        self.query_generation_step = QueryGenerationStep()
        self.document_retrieval_step = DocumentRetrievalStep()
        self.final_response_step = FinalResponseStep()
        self.steps = [
            self.plan_step,
            self.query_generation_step,
            self.document_retrieval_step,
            self.final_response_step
        ]

        self.vectorstore._initialize_vectorstore(llm=self.llm, embeddings=embeddings)

        for step in self.steps:
            step.llm = self.llm 
            step.query = self.query  
            self._steps_list.append(step)
            step_name = step.__class__.__name__
            self.steps[step_name] = step

    def run_from_specification(self):
        pass