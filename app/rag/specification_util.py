
from asyncio.log import logger
from langchain_ollama import ChatOllama
from config.openai_config import openapi_client
from model.rag_specification import LLMConfig
from model.rag_specification import LLMProvider
from rag.presets.preset_store import preset_store
from rag.rag_pipeline import Pipeline
from rag.presets.abstract.abstract_step_preset import AbstractStepPreset
from model.rag_specification import (
    RAGPipelineSpecification,
    PlanStepSpecification,
    QueryGenerationStepSpecification,
    RetrievalStepSpecification,
    FinalResponseStepSpecification
)

class SpecificationUtil:
    @staticmethod
    def _create_llm_from_config(self, llm_config: LLMConfig) -> ChatOllama:
        """Create an LLM instance from configuration"""
        if llm_config.provider == LLMProvider.OLLAMA:
            return ChatOllama(
                model=llm_config.model,
                temperature=llm_config.temperature,
                num_predict=llm_config.max_tokens
            )
        elif llm_config.provider == LLMProvider.OPENAI:
            return openapi_client(
                model=llm_config.model,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens
            )
        else:
            logger.warning(f"Unsupported LLM provider: {llm_config.provider}")
            return None
    
    @staticmethod
    def _create_preset_from_spec(self, preset_name: str, step_spec, query: str, llm) -> AbstractStepPreset:
        """Create and configure a preset instance from step specification"""
        try:
            preset = preset_store.get(
                name=preset_name,
                llm=llm,
                query=query
            )
            return preset
        except ValueError as e:
            logger.error(f"Failed to create preset '{preset_name}': {e}")
            logger.info(f"Available presets: {preset_store.list_presets()}")
            raise
            
    def create_default_rag_specification(self, query: str) -> RAGPipelineSpecification:
        """Create a basic RAG pipeline specification"""
        return RAGPipelineSpecification(
            input_query=query,
            global_llm_config=LLMConfig(
                provider=LLMProvider.OLLAMA,
                model="llama3.1",
                temperature=0.2,
                max_tokens=1000
            ),
            plan_step=PlanStepSpecification(
                use_anonymization=True,
                max_plan_steps=5
            ),
            query_generation_step=QueryGenerationStepSpecification(
                num_queries=3,
                query_optimization=False
            ),
            retrieval_step=RetrievalStepSpecification(
                k=5,
                retriever_types=["chunk"],
                use_mmr=False,
                rerank_results=False
            ),
            final_response_step=FinalResponseStepSpecification(
                response_style="comprehensive",
                include_sources=True,
                filter_irrelevant_content=True
            ),
            log_intermediate_results=True
        )
