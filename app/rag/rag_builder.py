from itertools import chain
import logging

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.schema import Document

from services.vector_store import VectorStore
from tools.query_augmentation_tool import QueryAugmentationTool
from tools.planning_tool import PlanningTool

from model.response_model import RAGResponse
from model.relevant_content_model import RelevantContent
from model.anonymize_model import DeanonymizedPlan
from model.plan_model import Plan
from model.chain_specification_model import LangchainSpecification
from model.llm_response_model import LLMResponse, ResponseType

from prompts.prompt_manager import PromptManager, analysis_prompt

from rag.steps.abstract.abstract_rag_step import AbstractRagStep

from config.openai_config import openapi_client
import uuid

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
        self.vectorstore = vectorstore
        self.planning_tool = PlanningTool(llm=llm)
        self.query = query
        self.llm = llm
        self.plan_obj: Plan = None
        self.generated_queries: list[str] = []
        self.steps: list[AbstractRagStep] = []

    def addStep(self, step: AbstractRagStep):
        self.steps.append(step)

    def run(self) -> LLMResponse:
    