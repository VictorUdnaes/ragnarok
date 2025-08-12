from langchain_ollama import ChatOllama
from rag.presets.preset_store import preset_store
from rag.presets.abstract.abstract_step_preset import AbstractStepPreset
from model.llm_response_model import LLMResponse, ResponseType
from langchain_core.documents import Document
from typing import List
from logging import getLogger
from services.vector_store import VectorStore

logger = getLogger("DocumentRetrievalPreset")

@preset_store.register("default_document_retrieval_preset")
class DefaultDocumentRetrievalPreset(AbstractStepPreset):
    def __init__(self):
        super().__init__()

    def execute(self, params: dict) -> LLMResponse:
        logger.info("Starting document retrieval preset execution")
        docs = self.retrieve_documents(
            queries=params.get("queries", []),
            chosen_retrievers=params.get("chosen_retrievers", []),
            k=params.get("k", 5)
        )

        self.response = LLMResponse(
            step_name="Document Retrieval",
            response_type=ResponseType.LIST,
            data=docs,
        )

        return self.response

    def retrieve_documents(self, queries: list[str], chosen_retrievers: list[str], k: int) -> list[Document]:
        print(f"Retrieving documents for queries: {queries}")
        docs: list[Document] = []
        allowed_retrievers = set(chosen_retrievers) & set(self.available_retrievers)
        for retriever in allowed_retrievers:
            retrieved = self.vectorstore.search_for_documents(
                queries=queries,
                retriever=retriever,
                k=k,
            )
            logger.info(f"Retrieved {len(retrieved)} '{retriever}' documents.")
            docs.extend(retrieved)
        logger.info(f"Retrieved a total of {len(docs)} documents from vectorstore.")
        return docs
