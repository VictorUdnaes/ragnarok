from fastapi import FastAPI, UploadFile, File, HTTPException
from services.rag_service import RagService
from prompts import multi_query_gen_prompt, analysis_prompt
from services.vector_store import VectorStore
from openai_config import openapi_client
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from utils.logger import logger
from typing import List


app = FastAPI()

vectorstore = VectorStore()
llm = openapi_client()
embeddings = OllamaEmbeddings(model="llama3.1")  # or another embedding model
vectorstore._initialize_vectorstore(llm=llm, embeddings=embeddings)

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "RAG Service is running"}

@app.post("/generate-response")
def generate_response(question: str):
    if not question:
        return {"error": "Question cannot be empty"}

    response = RagService() \
        .with_vectorstore(vectorstore) \
        .with_llm(model=llm, temperature=0) \
        .with_question(question=question) \
        .use_anonymized_planning() \
        .run(prompt=analysis_prompt)
    
    return response

@app.post("/embed-documents")
async def embed_documents(files: List[UploadFile] = File(...)):
    logger.info(f"Received {len(files)} files for embedding")
    
    if not files or len(files) == 0:
        logger.error("No files received for embedding.")
        raise HTTPException(status_code=400, detail="No files provided.")

    # Log each file received
    for i, file in enumerate(files):
        logger.info(f"File {i+1}: {file.filename}, size: {file.size if hasattr(file, 'size') else 'unknown'}")

    success_count = 0
    errors = []

    for file in files:
        try:
            logger.info(f"Processing file: {file.filename}")
            
            if not file.filename:
                logger.error("Filename is empty.")
                errors.append({"file": None, "error": "Filename cannot be empty"})
                continue

            # Check file type
            if not file.filename.lower().endswith('.pdf'):
                logger.error(f"Invalid file type for {file.filename}")
                errors.append({"file": file.filename, "error": "Only PDF files are allowed"})
                continue

            # Read file content to verify it's not empty
            content = await file.read()
            if len(content) == 0:
                logger.error(f"File {file.filename} is empty")
                errors.append({"file": file.filename, "error": "File is empty"})
                continue
            
            # Reset file position for processing
            await file.seek(0)
            
            # Process the file
            vectorstore.add_document_to_store(
                embedding_type="both",
                file=file
            )
            
            logger.info(f"Successfully embedded file: {file.filename}")
            success_count += 1

        except Exception as e:
            logger.error(f"Error embedding file {file.filename}: {str(e)}")
            logger.exception("Full traceback:")  # This will log the full stack trace
            errors.append({"file": file.filename, "error": str(e)})
        finally:
            # Ensure file is closed
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
    
    logger.info(f"Upload completed: {response}")
    return response