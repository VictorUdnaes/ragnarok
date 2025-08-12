import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_ollama import ChatOllama
from rag.rag_pipeline import Pipeline
from services.vector_store import VectorStore
from config.openai_config import openapi_client
from langchain_ollama import OllamaEmbeddings
from model.queries_from_plan import QueriesFromPlan
from model.llm_response_model import ResponseType
from rag.presets.query_generation_step.queries_from_plan_preset import QueriesFromPlanPreset
from rag.presets.retrieval_step.default_document_retrieval_preset import DefaultDocumentRetrievalPreset
from rag.presets.final_response_step.default_final_response_preset import DefaultFinalResponsePreset
from app.rag.presets.plan_step.anonymized_plan_preset import AnonymizedPlanPreset

llm = ChatOllama(model="llama3.1")

pipeline = Pipeline(
    llm=llm,
    query="Partiet Venstre ønsker å stoppe oljeutvinning i Norge",
    plan_step=AnonymizedPlanPreset(),
    query_generation_step=QueriesFromPlanPreset(),
    document_retrieval_step=DefaultDocumentRetrievalPreset(),
    final_response_step=DefaultFinalResponsePreset()
)

generated_plan_steps: list[str] = pipeline \
    .plan_step \
    .execute() \
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

generated_queries = pipeline                  \
    .query_generation_step                    \
    .execute(plan_steps=generated_plan_steps) \
    .data

print(generated_queries)

mock_queries = [
    "Hva er Venstres politikk for oljeutvinning?", 
    "Hvilke miljøpolitiske tiltak foreslår Venstre?",
    "Venstre klimapolitikk Norge"
]

vectorstore = VectorStore()
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore._initialize_vectorstore(llm=llm, embeddings=embeddings)

retrieved_docs = pipeline                     \
    .document_retrieval_step                  \
    .execute(
        vectorstore=vectorstore,
        queries=mock_queries,
        chosen_retrievers=["chunk", "quote"],
        k=5
    )                                         \
    .data

print(f"Retrieved {len(retrieved_docs)} documents.")

final_answer = pipeline \
    .final_response_step \
    .execute(
        documents=retrieved_docs,
        plan_steps=generated_plan_steps,
        generated_queries=generated_queries
    ) \
    .data

print(f"Final answer: {final_answer}")
