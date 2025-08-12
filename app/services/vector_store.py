from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from langchain.docstore.document import Document
from tools.embedding_tool import EmbeddingTool
from prompts.prompt_manager import remove_irrelevant_content_prompt
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from model.relevant_content_model import RelevantContent
from fastapi import UploadFile
from langchain_ollama import OllamaEmbeddings
import logging
import warnings
from json import dumps, loads

logger = logging.getLogger("ApplicationService")

class VectorStore:
    def __init__(self):
        self.chunk_vectorstore = None
        self.quote_vectorstore = None
        self.chunk_retriever = None
        self.quote_retriever = None
        self.embedder = EmbeddingTool()
        self.llm = None


    def _initialize_vectorstore(self, llm: ChatOpenAI, embeddings=OllamaEmbeddings(model="mxbai-embed-large")):
        try:            
            self.llm = llm

            # Initialize the vectorstore for storing chunks of text
            self.chunk_vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory="./app/chroma/chunk_chroma_db"
            )
            self.chunk_retriever = self.chunk_vectorstore.as_retriever(search_kwargs={"k": 5})

            #Initialize the vectorstore for storing quotes
            self.quote_vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory="./app/chroma/quote_chroma_db"
            )
            self.quote_retriever = self.quote_vectorstore.as_retriever(search_kwargs={"k": 5})

            test_results = self.chunk_vectorstore.similarity_search("test", k=1)
            if test_results:
                logger.info(f"[bold green]Vectorstore test successful - found {len(test_results)} test results[/bold green]")
            else:
                logger.warning("[bold yellow]Vectorstore test returned no results - database may be empty[/bold yellow]")

            logger.info("[bold green]Vectorstore initialized successfully.[/bold green]")
        except Exception as e:
            logger.error(f"Error initializing vectorstore: {e}")
            self.chunk_retriever = None
            raise

    def add_document_from_filepath(self, embedding_type: str, file_name: str):
        file_path = Path(__file__).parent.parent.parent / "app" / "documents" / file_name
        try:
            logger.info(f"Adding document to vectorstore from file: {file_path}")
            with open(file_path, "rb") as file:
                text_chunks = self.embedder.create_chunks_from_document(
                file=file,
                chunk_size=1000
            )

            text_quotes = self.embedder.create_chunks_from_pattern(
                file=file,
                pattern=r'Venstre (?:vil|ønsker)[^.]*\.'
            )
            
            self.chunk_vectorstore.add_documents(text_chunks)
            self.quote_vectorstore.add_documents(text_quotes)
        except Exception as e:
            logger.error(f"Error adding document from file {file_path} to vectorstore: {e}")
            raise

    def add_document_to_store(self, embedding_type: str, file: UploadFile):
        try:
            logger.info(f"Adding document to vectorstore: {file.filename}")
            text_chunks = self.embedder.create_chunks_from_document(
                file=file,
                chunk_size=1000
            )

            text_quotes = self.embedder.create_chunks_from_pattern(
                file=file,
                pattern=r'Venstre (?:vil|ønsker)[^.]*\.'
            )
           
            if embedding_type == "both":
                self.chunk_vectorstore.add_documents(text_chunks)
                self.quote_vectorstore.add_documents(text_quotes)

            elif embedding_type == "chunk":
                self.chunk_vectorstore.add_documents(text_chunks)

            elif embedding_type == "quote":
                self.quote_vectorstore.add_documents(text_quotes)

            logger.info(f"Document {file.filename} added successfully.")
        except Exception as e:
            logger.error(f"Error adding document {file.filename} to vectorstore: {e}")
            raise

    def search_for_documents(self, queries, retriever: str, k: int = 5) -> list[Document]:
        all_docs = []
        for i, query in enumerate(queries):
            try:
                if retriever == "chunk":
                    found_docs = self.chunk_retriever.invoke(query)
                else:
                    found_docs = self.quote_retriever.invoke(query)
                all_docs.extend(found_docs)
                logger.info(f"Query '{query}' returned {len(found_docs)} documents from {retriever} retriever")
                
            except Exception as e:
                logger.error(f"Error searching for documents with query '{query}': {e}")
                # Continue with next query instead of failing completely


        """# Apply relevance filtering to all documents at once
        if all_docs:
            try:
                relevant_docs = self.remove_irrelevant_content(queries, all_docs)
                return relevant_docs.relevant_content
            except Exception as e:
                logger.error(f"Error filtering relevant content: {e}")
                # Fall back to returning all documents if filtering fails
                return get_unique_union(all_docs)"""

        return self.get_unique_union(all_docs)

    def get_unique_union(self, documents: list[list]):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
                # Get unique documents
                unique_docs = list(set(flattened_docs))
                # Return
                return [loads(doc) for doc in unique_docs]
        except Exception as e:
            return []

    def remove_irrelevant_content(self, queries: list[str], retrieved_documents: list[Document]) -> str:
        keep_only_relevant_content_chain = PromptTemplate(
            template=remove_irrelevant_content_prompt,
            input_variables=["queries", "retrieved_documents"],
        ) | self.llm.with_structured_output(RelevantContent)
        
        relevant_content_obj: RelevantContent = keep_only_relevant_content_chain.invoke({
            "queries": queries,
            "retrieved_documents": retrieved_documents
        })

        return relevant_content_obj.relevant_content_as_string  
    
