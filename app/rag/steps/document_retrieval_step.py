from rag.steps.abstract.abstract_rag_step import AbstractRagStep
from model.llm_response_model import LLMResponse, ResponseType
from langchain_core.documents import Document
from typing import List, Type
from langchain_core.runnables import RunnableSequence
from prompts.prompt_manager import rerun_prompt
from logging import getLogger
from services.vector_store import VectorStore

logger = getLogger("DocumentRetrievalStep")

class DocumentRetrievalStep(AbstractRagStep):
    def __init__(self, vectorstore: VectorStore, queries: list[str], retrievers: list[str], k: int = 5):
        self.vectorstore = vectorstore
        self.queries = queries
        self.retrievers = retrievers
        self.k = k
        self.available_retrievers = ["chunk", "quote"]

    def execute(self):
        logger.info("Starting document retrieval step")
        docs = self.retrieve_documents()

        return LLMResponse(
            step_name="Document Retrieval",
            response_type=ResponseType.LIST,
            data=docs
        )
    
    def retry(self) -> LLMResponse:
        logger.info("Retrying document retrieval step execution")
        return self.execute()
    
    def retrieve_documents(self) -> list[Document]:
        docs = []
        allowed_retrievers = set(self.retrievers) & set(self.available_retrievers)
        for retriever in allowed_retrievers:
            retrieved = self.vectorstore.search_for_documents(
                queries=self.queries,
                retriever=retriever,
                k=self.k
            )
            logger.info(f"Retrieved {len(retrieved)} '{retriever}' documents.")
            docs.extend(retrieved)
        logger.info(f"Retrieved a total of {len(docs)} documents from vectorstore.")
        return docs