from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union
from enum import Enum

class LLMProvider(str, Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"

class LLMConfig(BaseModel):
    provider: LLMProvider = Field(default=LLMProvider.OLLAMA, description="LLM provider to use")
    model: str = Field(default="llama3.1", description="Model name")
    temperature: float = Field(default=0.2, description="Temperature for text generation")
    max_tokens: Optional[int] = Field(default=1000, description="Maximum tokens to generate")
    top_p: Optional[float] = Field(default=None, description="Top-p sampling parameter")
    frequency_penalty: Optional[float] = Field(default=None, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(default=None, description="Presence penalty")

class PlanStepSpecification(BaseModel):
    step_name: str = Field(default="Plan Generation", description="Human-readable name")
    preset_class: str = Field(default="anonymizer_preset", description="Preset class")
    use_anonymization: bool = Field(default=True, description="Whether to use anonymization")
    anonymization_style: str = Field(default="entities", description="Type of anonymization (entities, full, none)")
    max_plan_steps: int = Field(default=10, description="Maximum number of plan steps to generate")

class QueryGenerationStepSpecification(BaseModel):
    step_name: str = Field(default="Query Generation", description="Human-readable name")
    preset_class: str = Field(default="queries_from_plan_preset", description="Preset class")
    num_queries: int = Field(default=3, description="Number of queries to generate")
    query_optimization: bool = Field(default=False, description="Apply query optimization techniques")

class RetrievalStepSpecification(BaseModel):
    step_name: str = Field(default="Document Retrieval", description="Human-readable name")
    preset_class: str = Field(default="default_document_retrieval_preset", description="Preset class")

    # Retrieval specific parameters
    k: int = Field(default=5, description="Number of documents to retrieve per query")
    retriever_types: List[str] = Field(default=["chunk", "quote"], description="Types of retrievers to use")
    similarity_threshold: Optional[float] = Field(default=None, description="Minimum similarity threshold")
    use_mmr: bool = Field(default=False, description="Use Maximum Marginal Relevance")
    mmr_lambda: float = Field(default=0.5, description="MMR lambda parameter for diversity")
    rerank_results: bool = Field(default=False, description="Apply reranking to results")
    
class FinalResponseStepSpecification(BaseModel):
    step_name: str = Field(default="Final Response Generation", description="Human-readable name")
    preset_class: str = Field(default="default_final_response_preset", description="Preset class")
    response_style: str = Field(default="comprehensive", description="Style of response (comprehensive, concise, bullet_points)")
    include_sources: bool = Field(default=True, description="Include source citations in response")
    confidence_threshold: Optional[float] = Field(default=None, description="Minimum confidence threshold for answers")
    filter_irrelevant_content: bool = Field(default=True, description="Filter out irrelevant content before response generation")

class RAGPipelineSpecification(BaseModel):
    # Global settings
    global_llm_config: Optional[LLMConfig] = Field(default=None, description="Default LLM config for all steps")
    input_query: str = Field(description="The input question/query to process")
    execution_mode: str = Field(default="sequential", description="Execution mode (sequential, parallel)")
    
    # Step specifications
    plan_step: Optional[PlanStepSpecification] = Field(default_factory=PlanStepSpecification, description="Plan generation step config")
    query_generation_step: Optional[QueryGenerationStepSpecification] = Field(default_factory=QueryGenerationStepSpecification, description="Query generation step config")
    retrieval_step: Optional[RetrievalStepSpecification] = Field(default_factory=RetrievalStepSpecification, description="Retrieval step config")
    final_response_step: Optional[FinalResponseStepSpecification] = Field(default_factory=FinalResponseStepSpecification, description="Final response step config")
    
    # Pipeline-level settings
    enable_step_caching: bool = Field(default=False, description="Cache step results for replay")
    enable_corrections: bool = Field(default=True, description="Allow step corrections and retries")
    log_intermediate_results: bool = Field(default=True, description="Log intermediate step results")