from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings  # Changed to Ollama
from pathlib import Path
from utils.rag_util import get_unique_union
from utils.logger import logger  # Import the logger

class VectorStore:
    def __init__(self):
        self.retriever = None

    def _initialize_vectorstore(self, embeddings=OllamaEmbeddings(model="deepseek-r1:8b")):
        try:
            self.vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory="./chroma_db"
            )
            self.retriever = self.vectorstore.as_retriever()

            test_results = self.vectorstore.similarity_search("test", k=1)
            if test_results:
                logger.info(f"  |  Vectorstore initialized successfully")
            else:
                logger.warning("⚠ Vectorstore initialized but no documents found")
                logger.info("Use add_documents_to_vectorstore() to add documents first")

        except Exception as e:
            logger.error(f"Error initializing vectorstore: {e}", exc_info=True)
            self.retriever = None

    def retrieve_relevant_documents(self, queries, k: int = 5):
        logger.info("  |  Retrieving documents...")
        all_docs = []
        for i, query in enumerate(queries):
            try:
                docs = self.retriever.invoke(query)
                all_docs.extend(docs)
                logger.debug(f"  |  Query {i+1}: Retrieved {len(docs)} documents")
            except Exception as e:
                logger.error(f"Error retrieving for query {i+1}: {e}", exc_info=True)

        unique_docs = get_unique_union([all_docs])
        logger.info(f"  |  Found {len(unique_docs)} unique documents")
        return unique_docs

    def embedd_document(self, filename: str):
        logger.info(f"Embedding document: {filename}")
        try:
            current_dir = Path(__file__).parent
            path = current_dir.parent / 'documents' / filename

            loader = JSONLoader(
                file_path=path,
                jq_schema='.pages[].text',
                text_content=False
            )
            pages = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(pages)

            Chroma.from_documents(
                documents=chunks,
                embedding=OllamaEmbeddings(model="deepseek-r1:8b"),
                persist_directory="./chroma_db"
            )
            logger.info(f"Document {filename} embedded successfully")

        except Exception as e:
            logger.error(f"Error embedding document {filename}: {e}", exc_info=True)