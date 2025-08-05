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


        logger.info("[bold green]RAG Service initialized[/bold green]")
        self.vectorstore._initialize_vectorstore(llm=llm, embeddings=embeddings)

    def generate_anonymized_plan(self) -> LLMResponse:
        

    def generate_queries_with_anonymized_planning(self) -> LLMResponse:
        try:
            

            self.generated_queries = self.planning_tool.create_queries_from_plan(
                question=self.question,
                plan=self.plan_obj
            ).queries

            logger.info(f"Generated {len(self.generated_queries)} queries from plan:")
            for query in self.generated_queries:
                logger.info(f"[bold green][{query}][/bold green]")

            return LLMResponse(
                step_name="Queries from Plan",
                response_type=ResponseType.LIST,
                data=self.generated_queries
            )
        
        except Exception as e:
            logger.error(f"Error creating queries from plan: {e}")
            raise

    def generate_multiple_queries(self, prompt) -> LLMResponse:
        queries = QueryAugmentationTool.generate_multiple_queries(llm=self.llm, question=self.question, prompt=prompt)
        return LLMResponse(
            step_name="Multiple Queries Generation",
            response_type=ResponseType.LIST,
            data=queries
        )

    def retry_step(self, 
            step_name: str, 
            response_type: ResponseType, 
            specification: LangchainSpecification, 
            variables: dict) -> LLMResponse:
        response = specification \
            .build_chain(self.llm) \
            .invoke(variables)
        
        return LLMResponse(
            step_name=step_name,
            response_type=response_type,
            data=response,
        )

    def rerun_chain_with_feedback(self, 
            step_name: str, 
            response_type: ResponseType,                
            specification: LangchainSpecification, 
            original_input_variables: dict,
            original_response: LLMResponse,
            user_feedback: str) -> LLMResponse:
        
        enhanced_input = {
            **original_input_variables,  # Original variables
            "previous_attempt": original_response.data,
            "user_feedback": user_feedback
        }

        response = LangchainSpecification(
            input_variables=list(enhanced_input.keys()),
            prompt=rerun_prompt,
            format_object=chain.format_object
        ).build_chain(self.llm).invoke(enhanced_input)

        return LLMResponse(
            step_name=step_name,
            response_type=response_type,
            data=response,
        )
    
    def retrieve_documents(self, queries: list[str], retrievers: list[str], k: int = 5) -> list[Document]:
        docs = []
        allowed_retrievers = set(retrievers) & set(self.available_retrievers)
        for retriever in allowed_retrievers:
            retrieved = self.vectorstore.search_for_documents(
                queries=queries,
                retriever=retriever,
                k=k
            )
            logger.info(f"Retrieved {len(retrieved)} '{retriever}' documents.")
            docs.extend(retrieved)
        logger.info(f"Retrieved a total of {len(docs)} documents from vectorstore.")
        return docs
    
    def generate_final_response(self, documents: list[Document]) -> LLMResponse:
        final_chain = PromptTemplate(
                input_variables=["context", "plan", "original_question", "generated_queries_from_plan"],
                template=analysis_prompt
            ) | self.llm.with_structured_output(RAGResponse)

        answer = final_chain.invoke({
            "context": documents,
            "plan": self.plan_obj.steps,
            "original_question": self.query,
            "generated_queries_from_plan": self.generated_queries
        })
        logger.info(f"[bold blue]Final answer: {answer}[/bold blue]")
        logger.info("[bold green]RAG chain executed successfully.[/bold green]")
        
        return LLMResponse(
            step_name="Final Response",
            response_type=ResponseType.TEXT,
            data=answer
        )