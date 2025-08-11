from model.llm_response_model import LLMResponse
from langchain_core.documents import Document
from model.plan_model import Plan
from logging import getLogger
from rag.steps.final_response_step.default_final_response_preset import DefaultFinalResponsePreset

logger = getLogger("FinalResponseStep")


class FinalResponseStep:
    def __init__(self):
        self.default_final_response_preset = DefaultFinalResponsePreset()