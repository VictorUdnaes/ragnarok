from fastapi import FastAPI
from services.rag_service import RagChain
from app.prompts import multi_query_gen_prompt, analysis_prompt

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
        .with_llm("llama3.1", temperature=0) \
        .use_anonymized_planning() \
        .run(prompt=analysis_prompt)
    
    return response