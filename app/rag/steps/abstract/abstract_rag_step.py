from abc import ABC, abstractmethod
from typing import Any, Type
from model.llm_response_model import LLMResponse, ResponseType
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from model.chain_specification_model import LangchainSpecification
from pydantic import BaseModel
from prompts.prompt_manager import rerun_prompt
from config.openai_config import ChatOpenAI
from model.plan_model import Plan


class AbstractRagStep(ABC):
    def __init__(self, llm: ChatOpenAI):
        self.llm: ChatOpenAI = llm
        self.response: LLMResponse = None

    @abstractmethod
    def execute(self) -> LLMResponse:
        pass

    @abstractmethod
    def retry(self, response: LLMResponse) -> LLMResponse:
        pass
    
    @abstractmethod
    def rerun_with_correction(self, response: LLMResponse, correction: str) -> LLMResponse:
        pass

    def build_runnable_sequence(
            self, 
            input_variables: list[str], 
            prompt: str, 
            format_object
        ) -> RunnableSequence:
        chain = (
            PromptTemplate(
                input_variables=input_variables,
                template=prompt
            ) | self.llm.with_structured_output(Plan)
        )

        return chain 

    def build_correction_sequence(
            self, 
            original_input_variables: dict,
            original_prompt: str,
            format_object: Type[BaseModel],
            correction: str = None
        ) -> RunnableSequence:

        enhanced_input = {
            **original_input_variables,
            "original_prompt": original_prompt,
            "original_response": self.response.data,
            "correction": correction
        }

        chain = self.build_runnable_sequence(
            input_variables=list(enhanced_input.keys()),
            prompt=rerun_prompt,
            format_object=format_object
        )

        return chain
