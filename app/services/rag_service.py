from langchain_openai import OpenAIEmbeddings
from services.vector_store import VectorStore
from utils.rag_util import sanitize_response
from utils.logger import logger
from model.response_model import RAGResponse
from tools.query_augmentation_tool import QueryAugmentationTool
from openai_config import openapi_client
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from model.relevant_content_model import RelevantContent
from tools.planning_tool import PlanningTool
from langchain_core.prompts import PromptTemplate
from model.anonymize_model import DeanonymizedPlan
from langchain_openai import ChatOpenAI


class RagService:
    def __init__(self):
        self.vectorstore = VectorStore()
        openai_llm = None
        self.planning_tool = PlanningTool(llm=openai_llm)
        self.question = None
        self.plan_obj = None
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
        anonymized_question = self.planning_tool.anonymize_question(self.question)
        anonymized_plan = self.planning_tool.create_initial_plan(anonymized_question.question)
        self.plan_obj: DeanonymizedPlan = self.planning_tool.deanonymize_plan(
            plan=anonymized_plan.steps,
            mapping=anonymized_question.mapping
        )
        
        self.queries = self.planning_tool.create_queries_from_plan(
            question=self.question,
            plan=self.plan_obj.plan
        )

        logger.info(f"  |  Anonymized question: {anonymized_question.question}")
        logger.info(f"  |  Generated plan: {self.plan_obj.plan}")
        
        return self

    def use_multi_querying(self, prompt):
        self.queries = QueryAugmentationTool.generate_multiple_queries(llm=self.llm, question=self.question, prompt=prompt)
        logger.info(f"  |  Generated {len(self.queries)} query perspectives")

        return self

    def with_llm(self, model: ChatOpenAI, temperature=0):
        try:
            logger.info(f"  |  Initializing LLM with OpenAI model: {model.model_name}")
            self.llm = model
            embedding = OpenAIEmbeddings()
            self.vectorstore._initialize_vectorstore(llm=self.llm, embeddings=embedding)

        except Exception as e:
            logger.info(f"✗ Error initializing LLM: {e}")
            self.llm = None

        return self
    
    def with_vectorstore(self, vectorstore: VectorStore):
        if not isinstance(vectorstore, VectorStore):
            raise ValueError("Provided vectorstore is not an instance of VectorStore")
        
        self.vectorstore = vectorstore
        logger.info("  |  Using provided vectorstore")
        return self

    def run(self, prompt) -> RAGResponse:
        if not self.llm:
            return "Error: LLM not available"
        
        if not self.queries:
            logger.info("No queries generated, generating now...")
            self.use_multi_querying(self.question)

        try:
            retrieved_chunks: list[Document] = self.vectorstore \
                .search_for_documents(
                    queries=self.queries, 
                    retriever="chunk",
                    k=5
                )
            retrieved_quotes: list[Document] = self.vectorstore \
                .search_for_documents(
                    queries=self.queries, 
                    retriever="quote",
                    k=5
                )
            docs = retrieved_chunks + retrieved_quotes

            relevant_docs: RelevantContent = self.vectorstore.remove_irrelevant_content(
                query=self.question, 
                retrieved_documents=docs
            )

            final_chain = PromptTemplate(
                input_variables=["context", "plan", "question", "generated_queries"],
                template=prompt
            ) | self.llm.with_structured_output(RAGResponse)
     
            answer = final_chain.invoke({
                "context": relevant_docs.relevant_content,
                "plan": self.plan_obj.plan,
                "question": self.question,
                "generated_queries": self.queries
            })
            
            logger.info(f"<--  Final answer:\n{{{answer}}}")
            
            return answer
            
        except Exception as e:
            error_msg = f"Error executing RAG chain: {e}"
            logger.error(error_msg)
            return error_msg