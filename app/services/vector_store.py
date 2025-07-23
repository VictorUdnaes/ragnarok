from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings  # Changed to Ollama
from pathlib import Path
from langchain.docstore.document import Document
from utils.rag_util import get_unique_union
from utils.logger import logger  # Import the logger
from embedding.embedding_tool import EmbeddingTool  # Ensure OllamaEmbeddings is imported correctly

class VectorStore:
    def __init__(self):
        self.chunk_vectorstore = None
        self.quote_vectorstore = None
        self.chunk_retriever = None
        self.quote_retriever = None
        self.embedder = EmbeddingTool()


    def _initialize_vectorstore(self, embeddings=OllamaEmbeddings(model="llama3.1")):
        try:
            self.chunk_vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory="./chroma_db"
            )
            self.chunk_retriever = self.chunk_vectorstore.as_retriever()

            self.quote_vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory="./quote_chroma_db"
            )
            self.quote_retriever = self.quote_vectorstore.as_retriever()

            test_results = self.chunk_vectorstore.similarity_search("test", k=1)
            if test_results:
                logger.info(f"  |  Vectorstore initialized successfully")
            else:
                logger.warning("⚠ Vectorstore initialized but no documents found")
                logger.info("Use add_documents_to_vectorstore() to add documents first")

        except Exception as e:
            logger.error(f"Error initializing vectorstore: {e}", exc_info=True)
            self.chunk_retriever = None


    def search_for_documents(self, retriever: str, queries, k: int = 5) -> list[Document]:
        logger.info("  |  Retrieving quotes...")
        all_docs = []
        for i, query in enumerate(queries):
            try:
                found_docs = self.quote_retriever.invoke(query) if retriever == "chunk" else self.quote_retriever.invoke(query)
                all_docs.extend(found_docs)
                logger.debug(f"  |  Query {i+1}: Retrieved {len(found_docs)} documents")
                
                return get_unique_union([all_docs])
            
            except Exception as e:
                logger.error(f"Error retrieving for query {i+1}: {e}", exc_info=True)

        
    def add_document_to_store(self, filename: str):
        logger.info(f"Embedding document: {filename}")
        try:
            current_dir = Path(__file__).parent
            path: str = current_dir.parent / 'documents' / filename

            text_chunks = self.embedder.create_chunks_from_document(
                document_path=path,
                chunk_size=1000
            )

            self.chunk_vectorstore.add_documents(text_chunks)
            logger.info(f"Document {filename} embedded and added to vectorstore successfully")

        except Exception as e:
            logger.error(f"Error embedding document {filename}: {e}", exc_info=True)