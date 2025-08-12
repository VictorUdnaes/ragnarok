"""
Example usage of the RAG Pipeline with specifications
"""
from rag.rag_pipeline import Pipeline
from model.rag_specification import (
    RAGPipelineSpecification,
    LLMConfig,
    LLMProvider,
    PlanStepSpecification,
    QueryGenerationStepSpecification,
    RetrievalStepSpecification,
    FinalResponseStepSpecification,
)
from langchain_ollama import ChatOllama
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_default_rag_specification(query: str) -> RAGPipelineSpecification:
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

def create_custom_rag_specification(query: str) -> RAGPipelineSpecification:
    """Create a customized RAG pipeline specification with different settings per step"""
    return RAGPipelineSpecification(
        input_query=query,
        global_llm_config=LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3.1",
            temperature=0.1,
            max_tokens=1500
        ),
        plan_step=PlanStepSpecification(
            use_anonymization=False,  # Skip anonymization for faster execution
            max_plan_steps=3
        ),
        query_generation_step=QueryGenerationStepSpecification(
            num_queries=5,
            query_optimization=True
        ),
        retrieval_step=RetrievalStepSpecification(
            k=10,
            retriever_types=["chunk", "quote"],
            use_mmr=True,
            mmr_lambda=0.7,
            rerank_results=True
        ),
        final_response_step=FinalResponseStepSpecification(
            response_style="comprehensive",
            include_sources=True,
            filter_irrelevant_content=True
        ),
        enable_step_caching=False,
        log_intermediate_results=True
    )

def run_pipeline_example():
    """Example of running the RAG pipeline with a specification"""
    
    # Initialize the pipeline
    llm = ChatOllama(model="llama3.1", temperature=0.2)
    pipeline = Pipeline(llm=llm, query="")  # Query will be overridden by specification
    
    # Create specification
    query = "What are the main differences between the political programs?"
    spec = create_default_rag_specification(query)

    logger.info(f"Running RAG pipeline with query: {query}")
    
    # Execute pipeline
    results = pipeline.run_from_specification(spec)
    
    # Print results
    if "error" in results:
        logger.error(f"Pipeline execution failed: {results['error']}")
        return
    
    logger.info("Pipeline execution completed successfully!")
    logger.info(f"Execution order: {results['execution_order']}")
    
    # Print step results
    for step_name in results["execution_order"]:
        step_data = results["steps"][step_name]
        logger.info(f"\n{step_name.title()} Step Results:")
        logger.info(f"  Step name: {step_data['response'].step_name}")
        logger.info(f"  Response type: {step_data['response'].response_type}")
        logger.info(f"  Data preview: {str(step_data['data'])[:200]}...")
    
    # Get final answer
    if "final_response" in results["steps"]:
        final_answer = results["steps"]["final_response"]["data"]
        logger.info(f"\nFinal Answer: {final_answer}")
    
    return results

def run_custom_pipeline_example():
    """Example of running the RAG pipeline with custom configuration"""
    
    # Initialize the pipeline
    llm = ChatOllama(model="llama3.1", temperature=0.1)
    pipeline = Pipeline(llm=llm, query="")
    
    # Create custom specification
    query = "How does Venstre plan to improve education policy?"
    spec = create_custom_rag_specification(query)
    
    logger.info(f"Running custom RAG pipeline with query: {query}")
    
    # Execute pipeline
    results = pipeline.run_from_specification(spec)
    
    # Return results for further processing
    return results

if __name__ == "__main__":
    # Run basic example
    print("=" * 60)
    print("BASIC RAG PIPELINE EXAMPLE")
    print("=" * 60)
    run_pipeline_example()
    
    print("\n" + "=" * 60)
    print("CUSTOM RAG PIPELINE EXAMPLE")
    print("=" * 60)
    run_custom_pipeline_example()
