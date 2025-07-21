from langchain_ollama import OllamaEmbeddings
from services.vector_store import VectorStore
from utils.rag_util import sanitize_response
from utils.logger import logger
from app.model.response_model import RAGResponse
from app.tools.query_augmentation_tool import MultiQueryTool
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

class RagChain:
    def __init__(self):
        self.vectorstore = VectorStore()
        self.question = None
        self.queries = []
        self.llm = None
        logger.info("-->  Executing RAG Chain:")
    
    def with_question(self, question):
        if not question:
            raise ValueError("Question cannot be empty")
        
        logger.info(f"  |  Question: {question}")
        self.question = question
        return self
    
    def use_anonymized_planning(self):

        return self


    def use_multi_querying(self, prompt):
        self.queries = MultiQueryTool.generate_multiple_queries(llm=self.llm, question=self.question, prompt=prompt)
        logger.info(f"  |  Generated {len(self.queries)} query perspectives")

        return self
    

    def with_llm(self, model_name="deepseek-r1:8b", temperature=0):
        try:
            logger.info(f"  |  Initializing LLM with model: {model_name}")
            self.llm = ChatOllama(model=model_name, temperature=temperature)
            embedding = OllamaEmbeddings(model=model_name)
            self.vectorstore._initialize_vectorstore(embeddings=embedding)

        except Exception as e:
            logger.info(f"✗ Error initializing LLM: {e}")
            self.llm = None

        return self
        

    def run(self, prompt) -> RAGResponse:
        if not self.llm:
            return "Error: LLM not available"
        
        if not self.queries:
            logger.info("No queries generated, generating now...")
            self.use_multi_querying(self.question)
            
        try:
            retrieved_docs = self.vectorstore.retrieve_relevant_documents(queries=self.queries, k=5)
            
            context = "\n\n".join([doc.page_content for doc in retrieved_docs[:5]])  # Limit context
                        
            llm_prompt = ChatPromptTemplate.from_template(prompt)
            
            logger.info("  |  Generating final answer...")
            final_chain = llm_prompt | self.llm.with_structured_output(RAGResponse)
     
            answer = final_chain.invoke({
                "context": context,
                "question": self.question
            })
            
            logger.info(f"<--  Final answer:\n{{{answer}}}")
            
            return answer
            
        except Exception as e:
            error_msg = f"Error executing RAG chain: {e}"
            logger.error(error_msg)
            return error_msg
