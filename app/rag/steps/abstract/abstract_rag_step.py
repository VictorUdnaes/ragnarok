from abc import ABC, abstractmethod
from typing import Any, Type
from model.llm_response_model import LLMResponse, ResponseType
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from model.chain_specification_model import LangchainSpecification
from pydantic import BaseModel
from prompts.prompt_manager import rerun_prompt
from langchain_ollama import ChatOllama
from model.plan_model import Plan
from enum import Enum
import logging

logger = logging.getLogger("ApplicationService")

class AbstractRagStep(ABC):
    def __init__(self):
        self.llm: ChatOllama = None  # Type: ignore
        self.query: str = ""
        self.response: LLMResponse = None
        self.correctionDict: dict[str, str] = {}

    @abstractmethod
    def execute(self) -> LLMResponse:
        pass

    def retry(self) -> LLMResponse:
        logger.info("Retrying plan step execution")
        return self.execute()

    def rerun_with_correction(self, method_name: str, correction: str) -> LLMResponse:
        self.register_correction(method_name, correction)
        logger.info("Rerunning step with correction")
        return self.execute()

    def _execute_runnable_sequence_from_prompt(
            self, 
            input_variables: dict[str, str],
            prompt: str, 
            format_object: Type[BaseModel],
        ):
        chain = (
            PromptTemplate(
                input_variables=input_variables.keys(),
                template=prompt
            ) | self.llm.with_structured_output(format_object)
        )

        return chain.invoke(input_variables)

    def _execute_correction_sequence(
            self, 
            method_name: str,
            input_variables: dict[str, str],
            original_prompt: str,
            format_object: Type[BaseModel],
        ):

        corrected_input_variables = {
            "original_prompt": original_prompt,
            "original_response": self.response, 
            "correction": self.correctionDict[method_name],
            "original_input_variables": str(input_variables)
        }

        chain = (
            PromptTemplate(
                input_variables=corrected_input_variables.keys(),
                template=rerun_prompt
            ) | self.llm.with_structured_output(format_object)
        )

        return chain.invoke(corrected_input_variables)

    def execute_runnable_sequence(
            self, 
            method_name: str, 
            input_variables: dict, 
            prompt: str, 
            format_object: Type[BaseModel]
        ):
        try:
            response = None
            if self.should_rerun(method_name):
                logger.info(f"Executing {method_name} with correction")
                response = self._execute_correction_sequence(method_name, input_variables, prompt, format_object)
            else:
                logger.info(f"Executing {method_name} normally")
                response = self._execute_runnable_sequence_from_prompt(input_variables, prompt, format_object)
            self._clear_correction(method_name)
            
            return response
        except Exception as e:
            logger.error(f"Error in {method_name}: {e}")
            # Could implement fallback logic here
            raise

    def register_correction(self, method_name: str, correction: str):
        self.correctionDict[method_name] = correction
    
    def _clear_correction(self, method_name: str):
        self.correctionDict.pop(method_name, None)

    def should_rerun(self, method_name: str) -> bool:
        return method_name in self.correctionDict