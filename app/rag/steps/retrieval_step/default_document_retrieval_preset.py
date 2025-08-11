from rag.steps.abstract.abstract_step_preset import AbstractStepPreset
from model.llm_response_model import LLMResponse, ResponseType
from langchain_core.documents import Document
from typing import List
from logging import getLogger

logger = getLogger("DocumentRetrievalPreset")


class DefaultDocumentRetrievalPreset(AbstractStepPreset):
    def __init__(self):
        super().__init__()
        self.queries: list[str] = []
        self.chosen_retrievers: List[str] = []
        self.k: int = 5

    def execute(self) -> LLMResponse:
        logger.info("Starting document retrieval preset execution")
        docs = self.retrieve_documents()
        return LLMResponse(
            step_name="Document Retrieval",
            response_type=ResponseType.LIST,
            data=docs,
        )

    def retrieve_documents(self) -> list[Document]:
        print(f"Retrieving documents for queries: {self.queries}")
        docs: list[Document] = []
        allowed_retrievers = set(self.chosen_retrievers) & set(self.available_retrievers)
        for retriever in allowed_retrievers:
            retrieved = self.vectorstore.search_for_documents(
                queries=self.queries,
                retriever=retriever,
                k=self.k,
            )
            logger.info(f"Retrieved {len(retrieved)} '{retriever}' documents.")
            docs.extend(retrieved)
        logger.info(f"Retrieved a total of {len(docs)} documents from vectorstore.")
        return docs
