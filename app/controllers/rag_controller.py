from fastapi import FastAPI
from services.rag_service import RagChain
from utils.promps import MULTI_QUERY_GEN_PROMPT, POLITICAL_ANALYSIS_PROMPT

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "RAG Service is running"}

@app.post("/generate-response")
def generate_response(question: str):
    if not question:
        return {"error": "Question cannot be empty"}

    response = RagChain() \
        .with_question(question=question) \
        .with_llm("deepseek-r1:8b") \
        .with_multi_querying(prompt=MULTI_QUERY_GEN_PROMPT) \
        .run(prompt=POLITICAL_ANALYSIS_PROMPT)
    
    return response