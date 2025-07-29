from services.vector_store import VectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from app.tools.embedding_tool import EmbeddingTool
from pathlib import Path
from app.openai_config import openapi_client

vectorstore = VectorStore()
llm = openapi_client()
embedding = OpenAIEmbeddings()
vectorstore._initialize_vectorstore(llm=llm, embeddings=embedding)

print("Adding documents to vectorstore...")
vectorstore.add_document_to_store(
    embedding_type="both",
    filename="venstre-prinsipprogram-2020.pdf"
)
vectorstore.add_document_to_store(
    embedding_type="both",
    filename="venstre-stortingsprogram-2025.pdf"
)

print("Documents added to vectorstore successfully.")
