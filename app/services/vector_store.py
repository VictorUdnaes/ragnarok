from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from langchain.docstore.document import Document
from utils.rag_util import get_unique_union
from utils.logger import logger  # Import the logger
from tools.embedding_tool import EmbeddingTool
from prompts import remove_irrelevant_content_prompt
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from model.relevant_content_model import RelevantContent
from fastapi import UploadFile
from langchain_ollama import OllamaEmbeddings

class VectorStore:
    def __init__(self):
        self.chunk_vectorstore = None
        self.quote_vectorstore = None
        self.chunk_retriever = None
        self.quote_retriever = None
        self.embedder = EmbeddingTool()
        self.llm = None


    def _initialize_vectorstore(self, llm: ChatOpenAI, embeddings=OllamaEmbeddings(model="llama3.1")):
        try:            
            self.llm = llm

            # Initialize the vectorstore for storing chunks of text
            self.chunk_vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory="./chroma/chunk_chroma_db"
            )
            self.chunk_retriever = self.chunk_vectorstore.as_retriever()

            #Initialize the vectorstore for storing quotes
            self.quote_vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory="./chroma/quote_chroma_db"
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

    # Endre så den tar embedding pattern som parameter
    def add_document_to_store(self, embedding_type: str, file: UploadFile):
        logger.info(f"Embedding document: {file.filename}")
        try:
            text_chunks = self.embedder.create_chunks_from_document(
                file=file,
                chunk_size=1000
            )

            text_quotes = self.embedder.create_chunks_from_pattern(
                file=file,
                pattern=r'Venstre (?:vil|ønsker)[^.]*\.'
            )
            self.quote_vectorstore.add_documents(text_quotes)

            if embedding_type == "both":
                self.chunk_vectorstore.add_documents(text_chunks)
                self.quote_vectorstore.add_documents(text_quotes)

            elif embedding_type == "chunk":
                self.chunk_vectorstore.add_documents(text_chunks)

            elif embedding_type == "quote":
                self.chunk_vectorstore.add_documents(text_quotes)

            logger.info(f"Document {file.filename} embedded and added to vectorstore successfully")

        except Exception as e:
            logger.error(f"Error embedding document {file.filename}: {e}", exc_info=True)


    def search_for_documents(self, retriever: str, queries, k: int = 5) -> list[Document]:
        logger.info("  |  Retrieving quotes...")
        all_docs = []
        for i, query in enumerate(queries):
            try:
                found_docs = self.quote_retriever.invoke(query) if retriever == "chunk" else self.quote_retriever.invoke(query)
                relevant_docs = self.remove_irrelevant_content(query, found_docs)
                all_docs.extend(relevant_docs.relevant_content)
                logger.debug(f"  |  Query {i+1}: Retrieved {len(found_docs)} documents")
                
            except Exception as e:
                logger.error(f"Error retrieving for query {i+1}: {e}", exc_info=True)

        return get_unique_union(all_docs)


    def remove_irrelevant_content(self, query: str, retrieved_documents: list[Document]) -> RelevantContent:
        keep_only_relevant_content_chain = PromptTemplate(
            template=remove_irrelevant_content_prompt,
            input_variables=["query", "retrieved_documents"],
        ) | self.llm.with_structured_output(RelevantContent)
        
        relevant_content_obj: RelevantContent = keep_only_relevant_content_chain.invoke({
            "query": query,
            "retrieved_documents": retrieved_documents
        })

        relevant_content_obj.relevant_content = "".join(relevant_content_obj)

        return relevant_content_obj \
            .relevant_content \
            .replace('"', '\\"') \
            .replace("'", "\\'")