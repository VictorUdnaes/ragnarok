from fastapi import FastAPI
from services.rag_service import RagChain
from app.utils.prompts import multi_query_gen_prompt, analysis_prompt

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
        .with_llm("deepseek-r1:8b", temperature=0) \
        .use_multi_querying(prompt=multi_query_gen_prompt) \
        .run(prompt=analysis_prompt)
    
    return response