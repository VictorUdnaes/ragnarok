from fastapi import FastAPI, UploadFile, File, HTTPException
from services.rag_service import RagService
from prompts import multi_query_gen_prompt, analysis_prompt
from services.vector_store import VectorStore
from openai_config import openapi_client, openapi_embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from typing import List
from rich_logging_setup import RichLoggingSetup, RichLoggingMiddleware
import logging

rich_logging_setup = RichLoggingSetup()
app = FastAPI()
app.add_middleware(RichLoggingMiddleware)
rich_logging_setup.log_startup_banner()

logger = logging.getLogger("ApplicationService")

vectorstore = VectorStore()
llm = openapi_client()
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

@app.get("/health")
def health_check():
    logger.info("Health check endpoint called.")
    return {"status": "healthy", "message": "RAG Service is running"}

@app.post("/generate-response")
def generate_response(question: str):
    logger.info(f"\nGenerate response endpoint called with question: {question}")

    if not question:
        logger.error("Question cannot be empty.")
        return {"error": "Question cannot be empty"}

    try:
        response = RagService() \
            .with_vectorstore(vectorstore) \
            .with_llm(model=llm, embeddings=embeddings, temperature=0) \
            .with_question(question=question) \
            .with_anonymized_planning() \
            .run(prompt=analysis_prompt)

        return response

    except Exception as e:
        logger.exception("Error generating response.")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/embed-documents")
async def embed_documents(files: List[UploadFile] = File(...)):
    vectorstore._initialize_vectorstore(llm=llm, embeddings=embeddings)

    if not files or len(files) == 0:
        logger.error("No files provided for embedding.")
        raise HTTPException(status_code=400, detail="No files provided.")

    success_count = 0
    errors = []

    for file in files:
        try:
            logger.info(f"Processing file: {file.filename}")

            if not file.filename:
                logger.error("Filename is empty.")
                errors.append({"file": None, "error": "Filename cannot be empty"})
                continue

            if not file.filename.lower().endswith('.pdf'):
                logger.error(f"Invalid file type for {file.filename}")
                errors.append({"file": file.filename, "error": "Only PDF files are allowed"})
                continue

            content = await file.read()
            if len(content) == 0:
                logger.error(f"File {file.filename} is empty.")
                errors.append({"file": file.filename, "error": "File is empty"})
                continue

            await file.seek(0)

            vectorstore.add_document_to_store(
                embedding_type="both",
                file=file
            )

            logger.info(f"Successfully embedded file: {file.filename}")
            success_count += 1

        except Exception as e:
            logger.error(f"Error embedding file {file.filename}: {str(e)}")
            errors.append({"file": file.filename, "error": str(e)})
        finally:
            if hasattr(file, 'file') and file.file:
                file.file.close()

    status = "success" if success_count == len(files) and not errors else "partial_success" if success_count > 0 else "failed"

    response = {
        "status": status,
        "message": f"{success_count} out of {len(files)} document(s) embedded successfully",
        "total_files": len(files),
        "successful": success_count,
        "errors": errors if errors else None
    }

    logger.info(f"Embedding process completed with status: {status}")
    return response