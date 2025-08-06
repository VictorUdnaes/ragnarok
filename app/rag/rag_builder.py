import logging
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from services.vector_store import VectorStore
from tools.planning_tool import PlanningTool
from model.plan_model import Plan
from rag.steps.abstract.abstract_rag_step import AbstractRagStep
from config.openai_config import openapi_client

logger = logging.getLogger("ApplicationService")

class RagBuilder:
    def __init__(
            self, 
            vectorstore: VectorStore, 
            embeddings: OllamaEmbeddings,
            llm: ChatOpenAI, 
            query: str
        ):
        self.available_retrievers: list[str] = ["chunk", "quote"]
        self.vectorstore = vectorstore._initialize_vectorstore(llm=llm, embeddings=embeddings)
        self.planning_tool = PlanningTool(llm=llm)
        self.query = query
        self.llm = llm
        self.plan_obj: Plan = None
        self.generated_queries: list[str] = []
        self.steps: dict[str, AbstractRagStep] = {}
        self._steps_list: list[AbstractRagStep] = []

    def addStep(self, step: AbstractRagStep):
        step.llm = self.llm  # Ensure the step has access to the LLM
        step.query = self.query  # Ensure the step has access to the query
        self._steps_list.append(step)
        step_name = step.__class__.__name__
        self.steps[step_name] = step

    def get_steps_list(self) -> list[AbstractRagStep]:
        """Get the list of steps for iteration"""
        return self._steps_list
    