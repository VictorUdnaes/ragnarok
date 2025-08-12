import logging
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from config.openai_config import openapi_client
from rag.presets.abstract.abstract_step_preset import AbstractStepPreset
from rag.presets.preset_store import preset_store

logger = logging.getLogger("ApplicationService")

class Pipeline():
    def __init__(
            self, 
            llm: ChatOllama, 
            query: str,
            plan_preset: AbstractStepPreset,
            query_generation_preset: AbstractStepPreset,
            document_retrieval_preset: AbstractStepPreset,
            final_response_preset: AbstractStepPreset
        ):
        self.query = query
        self.llm = llm
        # RAG Steps
        self.plan_step = plan_preset
        self.query_generation_step = query_generation_preset
        self.document_retrieval_step = document_retrieval_preset
        self.final_response_step = final_response_preset

        for preset in [self.plan_step, self.query_generation_step, self.document_retrieval_step, self.final_response_step]:
            preset.setLLM(self.llm)
            preset.setQuery(self.query)
