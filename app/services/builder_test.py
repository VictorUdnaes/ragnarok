import sys
import os

# Add the app directory to Python path so imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_ollama import ChatOllama
from rag.rag_builder import RagBuilder
from services.vector_store import VectorStore
from config.openai_config import openapi_client
from langchain_ollama import OllamaEmbeddings
from rag.steps.plan_step import PlanStep
from model.llm_response_model import ResponseType

vectorstore = VectorStore()
llm = ChatOllama(model="llama3.1")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

execution_chain = RagBuilder(
    vectorstore=vectorstore,
    embeddings=embeddings,
    llm=llm,
    query="Partiet Venstre ønsker å stoppe oljeutvinning i Norge"
)

plan_step = PlanStep()
execution_chain.addStep(step=plan_step)

first_response = execution_chain.steps["PlanStep"].execute()

print(first_response.data)

corrected_response = execution_chain.steps["PlanStep"].rerun_with_correction(
    method_name="anonymize_question",
    correction="You didn't anonymize correctly, 'Norge' should be replaced"
)