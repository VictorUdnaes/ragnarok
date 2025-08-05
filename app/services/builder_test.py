from app.rag.rag_builder import RagBuilder
from services.vector_store import VectorStore
from config.openai_config import openapi_client
from langchain_ollama import OllamaEmbeddings
vectorstore = VectorStore()
llm = openapi_client()
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

execution_chain = RagBuilder(
    vectorstore=vectorstore,
    embeddings=embeddings,
    llm=llm,
    query="Partiet Venstre ønsker å stoppe oljeutvinning i Norge"
)

queries = execution_chain.generate_queries_with_anonymized_planning()
docs = execution_chain.retrieve_documents(
    queries=queries,
    retrievers=["chunk", "quote"]
)
final_response = execution_chain.generate_final_response(documents=docs)


