from model.llm_response_model import LLMResponse
from logging import getLogger
from rag.steps.retrieval_step.default_document_retrieval_preset import DefaultDocumentRetrievalPreset

logger = getLogger("DocumentRetrievalStep")

class DocumentRetrievalStep:
    def __init__(self):
        self.default_retrieval_preset = DefaultDocumentRetrievalPreset()
