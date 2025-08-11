import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import ChatOllama
from rag.rag_pipeline import Pipeline
from services.vector_store import VectorStore
from config.openai_config import openapi_client
from langchain_ollama import OllamaEmbeddings
from rag.steps.plan_step.plan_step import PlanStep
from rag.steps.retrieval_step.document_retrieval_step import DocumentRetrievalStep
from model.queries_from_plan import QueriesFromPlan
from rag.steps.query_generation_step.query_generation_step import QueryGenerationStep
from model.llm_response_model import ResponseType


vectorstore = VectorStore()
llm = ChatOllama(model="llama3.1")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

pipeline = Pipeline(
    vectorstore=vectorstore,
    embeddings=embeddings,
    llm=llm,
    query="Partiet Venstre ønsker å stoppe oljeutvinning i Norge"
)

generated_plan_steps: list[str] = pipeline \
    .plan_step                             \
    .anonymizer_preset                     \
    .execute()                             \
    .data

print(generated_plan_steps)

"""
corrected_response = pipeline \
    .plan_step \
    .anonymizer_preset \
    .rerun_with_correction(
        method_name="anonymize_question",
        correction="You didn't anonymize correctly, 'Norge' should be replaced"
    ) \
    .data
"""

generated_queries = pipeline                            \
    .query_generation_step                              \
    .queries_from_plan_preset                           \
    .execute(generated_plan_steps=generated_plan_steps) \
    .data

print(generated_queries)

mock_queries = [
    "Hva er Venstres politikk for oljeutvinning?", 
    "Hvilke miljøpolitiske tiltak foreslår Venstre?",
    "Venstre klimapolitikk Norge"
]

retrieved_docs = pipeline                     \
    .document_retrieval_step                  \
    .default_retrieval_preset                 \
    .execute(
        queries=mock_queries,
        chosen_retrievers=["chunk", "quote"],
        k=5
    )                                         \
    .data

print(f"Retrieved {len(retrieved_docs)} documents.")
