from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Type
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from model.llm_response_model import ResponseType

class LangchainSpecification(BaseModel):
    input_variables: list[str]
    prompt: str
    format_object: Type[BaseModel]


    def build_chain(self, llm: ChatOpenAI) -> RunnableSequence:
        """Builds a chain using the provided input variables, prompt, and format object."""
        return (
            PromptTemplate(
                input_variables=self.input_variables,
                template=self.prompt
            ) | llm.with_structured_output(self.format_object)
        )