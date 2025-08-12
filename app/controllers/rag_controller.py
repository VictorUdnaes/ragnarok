from fastapi import FastAPI, UploadFile, File, HTTPException
from rag.rag_pipeline import Pipeline
from services.vector_store import VectorStore
from config.openai_config import openapi_client, openapi_embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from typing import List
from config.rich_logging_setup import RichLoggingSetup, RichLoggingMiddleware
import logging
from prompts.prompt_manager import analysis_prompt
from pydantic import BaseModel
from rag.presets.preset_store import preset_store
from model.execute_step_request import ExecuteStepRequest
# Dette er ikke ideelt. preset klassene må importeres selv om de ikke brukes, ellers blir de ikke lastet inn på runtime og da fungerer ikke 
# @preset_store.register() metoden.
from rag.presets.plan_step.anonymized_plan_preset import AnonymizedPlanPreset
from rag.presets.query_generation_step.queries_from_plan_preset import QueriesFromPlanPreset
from rag.presets.retrieval_step.default_document_retrieval_preset import DefaultDocumentRetrievalPreset
from rag.presets.final_response_step.default_final_response_preset import DefaultFinalResponsePreset
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.routing import APIRoute
from typing import List
import logging


# Setup logging
rich_logging_setup = RichLoggingSetup()
logger = logging.getLogger("ApplicationService")

# Initialize FastAPI app with rich logging middleware
app = FastAPI()
app.add_middleware(RichLoggingMiddleware)

class InitializationRequest(BaseModel):
    llm_model: str = "llama3.1"
    embeddings_model: str = "mxbai-embed-large"
    temperature: float = 0.2

class RAGController:
    def __init__(self):
        self.router = APIRouter(prefix="/api/v1", tags=["RAG"])
        self.vectorstore = VectorStore()
        self.llm = None
        self.embeddings = None
        self.logger = logging.getLogger("RAGController")
        
        # Register all routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup all API routes"""
        self.router.add_api_route("/health", self.health_check, methods=["GET"])
        self.router.add_api_route("/initialize", self.initialize, methods=["POST"])
        # Embedding
        self.router.add_api_route("/embed-documents", self.embed_documents, methods=["POST"])
        self.router.add_api_route("/delete-document", self.delete_document, methods=["DELETE"])
        # RAG
        self.router.add_api_route("/execute-step", self.execute_step, methods=["POST"])
        self.router.add_api_route("/generate-response", self.generate_response, methods=["POST"])

    async def initialize(self, request: InitializationRequest):
        """Initialize RAG components"""
        self.logger.info(f"Initializing with LLM model: {request.llm_model}, Embeddings model: {request.embeddings_model}")

        self.llm = ChatOllama(model=request.llm_model, temperature=request.temperature)
        self.embeddings = OllamaEmbeddings(model=request.embeddings_model)
        
        return {"status": "initialized", "llm_model": request.llm_model}


    async def execute_step(self, request: ExecuteStepRequest):
        """Execute a single RAG step with context"""
        try:
            preset = preset_store.get(
                name=request.preset,
                llm=self.llm,
                vectorstore=self.vectorstore,
            )
            spec = request.deserialize(preset.get_spec_class())
            result = preset.execute(spec=spec)

            return {
                "status": "success",
                "step_name": request.step_name, 
                "preset": request.preset,
                "result": result.data
            }
        except Exception as e:
            self.logger.error(f"Step execution failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


    async def generate_response(self, request):
        """Generate response using full RAG pipeline"""
        self.logger.info(f"Generate response endpoint called with question: {request.question}")

        if not request.question:
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        try:
            llm = ChatOllama(model="llama3.1", temperature=0.2)
            pipeline = Pipeline(llm=llm, query=request.question)
            spec = pipeline.create_default_rag_specification(request.question)
            
            results = pipeline.run_from_specification(spec)
            return results["steps"]["final_response"]["data"]

        except Exception as e:
            self.logger.exception("Error generating response.")
            raise HTTPException(status_code=500, detail="Internal Server Error")
        

    async def embed_documents(self, files: List[UploadFile] = File(...)):
        """Embed documents into vector store"""
        self.vectorstore._initialize_vectorstore(llm=self.llm, embeddings=self.embeddings)

        if not files:
            raise HTTPException(status_code=400, detail="No files provided.")

        success_count = 0
        errors = []

        for file in files:
            try:
                self.logger.info(f"Processing file: {file.filename}")

                if not file.filename or not file.filename.lower().endswith('.pdf'):
                    errors.append({"file": file.filename, "error": "Only PDF files are allowed"})
                    continue

                content = await file.read()
                if len(content) == 0:
                    errors.append({"file": file.filename, "error": "File is empty"})
                    continue

                await file.seek(0)
                self.vectorstore.add_document_to_store(embedding_type="both", file=file)
                success_count += 1

            except Exception as e:
                self.logger.error(f"Error embedding file {file.filename}: {str(e)}")
                errors.append({"file": file.filename, "error": str(e)})

        return {
            "status": "success" if success_count == len(files) else "partial_success",
            "successful": success_count,
            "total_files": len(files),
            "errors": errors if errors else None
        }


    async def delete_document(self, filename: str):
        """Delete documents by filename"""
        try:
            result = self.vectorstore.delete(where={"source": filename})
            return {"status": "success", "deleted": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


    async def health_check(self):
        """Health check endpoint"""
        self.logger.info("Health check endpoint called.")
        return {"status": "healthy", "message": "RAG Service is running"}


# Create controller instance and setup app
def create_app() -> FastAPI:
    app = FastAPI(title="RAG API", version="1.0.0")
    
    # Setup logging
    rich_logging_setup = RichLoggingSetup()
    rich_logging_setup.log_startup_banner()
    app.add_middleware(RichLoggingMiddleware)
    
    # Create and register controller
    rag_controller = RAGController()
    app.include_router(rag_controller.router)
    
    return app

# Create the app instance
app = create_app()
