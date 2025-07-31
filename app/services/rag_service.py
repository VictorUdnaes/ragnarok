import logging
from langchain_openai import OpenAIEmbeddings
from services.vector_store import VectorStore
from model.response_model import RAGResponse
from tools.query_augmentation_tool import QueryAugmentationTool
from config.openai_config import openapi_client
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from model.relevant_content_model import RelevantContent
from tools.planning_tool import PlanningTool
from langchain_core.prompts import PromptTemplate
from model.anonymize_model import DeanonymizedPlan
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from model.plan_model import Plan
from prompts.prompt_manager import PromptManager


logger = logging.getLogger("ApplicationService")

class RagService:
    def __init__(self):
        self.vectorstore = VectorStore()
        self.planning_tool = None
        self.question = None
        self.plan_obj = None
        self.queries = []
        self.llm = None
        self.should_use_anonymized_planning = False
        logger.info("[bold green]RAG Service initialized[/bold green]")

    def with_llm(self, model: ChatOpenAI, embeddings: OllamaEmbeddings, temperature=0):
        try:
            logger.info(f"[bold green]Configuring LLM with model: {model.model_name}[/bold green]")
            self.llm = model
            self.planning_tool = PlanningTool(llm=self.llm)
            self.vectorstore._initialize_vectorstore(llm=self.llm, embeddings=embeddings)
        except Exception as e:
            logger.error(f"Error configuring LLM: {e}")
            self.llm = None
        return self

    def with_vectorstore(self, vectorstore: VectorStore):
        if not isinstance(vectorstore, VectorStore):
            logger.error("Provided vectorstore is not an instance of VectorStore.")
            raise ValueError("Provided vectorstore is not an instance of VectorStore")

        self.vectorstore = vectorstore
        return self

    def with_anonymized_planning(self):
        self.should_use_anonymized_planning = True
        logger.info("[bold green]Anonymized planning enabled.[/bold green]")
        return self

    def with_question(self, question):
        if not question:
            logger.error("Question cannot be empty.")
            raise ValueError("Question cannot be empty")

        self.question = question
        logger.info(f"[bold green]Question set:[/bold green] {question}")
        return self

    def create_queries_from_plan(self):
        try:
            anonymized_question_obj = self.planning_tool.anonymize_question(self.question)
            anonymized_plan = self.planning_tool.create_initial_plan(anonymized_question_obj.anonymized_question)
            
            self.plan_obj: Plan = self.planning_tool.deanonymize_plan(
                plan=anonymized_plan.steps,
                mapping=anonymized_question_obj.mapping
            )

            logger.info(f"Plan created with {len(self.plan_obj.steps)} steps:")
            for step in enumerate(self.plan_obj.steps):
                logger.info(f"[bold green][{step}][/bold green]")

            queries_obj = self.planning_tool.create_queries_from_plan(
                question=self.question,
                plan=self.plan_obj
            )

            self.queries = queries_obj.queries

            logger.info(f"Generated {len(self.queries)} queries from plan:")
            for query in self.queries:
                logger.info(f"[bold green][{query}][/bold green]")
        except Exception as e:
            logger.error(f"Error creating queries from plan: {e}")
            raise

    def generate_multiple_queries(self, prompt):
        self.queries = QueryAugmentationTool.generate_multiple_queries(llm=self.llm, question=self.question, prompt=prompt)


    def run(self, prompt) -> RAGResponse:
        if not self.llm:
            return "Error: LLM not available"

        try:
            logger.info("[bold yellow]<-- Executing RAG chain -->[/bold yellow]")
            # Create queries from plan if anonymized planning is enabled
            if self.should_use_anonymized_planning:
                self.create_queries_from_plan()
            else:
                self.queries = [self.question]
            
            # Retrieve documents from vectorstore using the chunk and quote retrievers
            retrieved_chunks: list[Document] = self.vectorstore \
                .search_for_documents(
                    queries=self.queries, 
                    retriever="chunk",
                    k=5
                )
            logger.info(f"Retrieved {len(retrieved_chunks)} chunk documents.")

            retrieved_quotes: list[Document] = self.vectorstore \
                .search_for_documents(
                    queries=self.queries, 
                    retriever="quote",
                    k=5
                )
            logger.info(f"Retrieved {len(retrieved_quotes)} quote documents.")
            docs = retrieved_chunks + retrieved_quotes
            logger.info(f"Retrieved a total of {len(docs)} documents from vectorstore.")

            # Remove irrelevant content from retrieved documents
            """
            relevant_content_as_string: str = self.vectorstore.remove_irrelevant_content(
                queries=self.queries, 
                retrieved_documents=docs
            )
            logger.info("Content retrieved: %s", relevant_content_as_string)
            """

            # Generate final answer using the LLM and the provided prompt
            final_chain = PromptTemplate(
                input_variables=["context", "plan", "original_question", "generated_queries_from_plan"],
                template=prompt
            ) | self.llm.with_structured_output(RAGResponse)
     
            answer = final_chain.invoke({
                "context": docs,
                "plan": self.plan_obj.steps,
                "original_question": self.question,
                "generated_queries_from_plan": self.queries
            })
            logger.info(f"[bold blue]Final answer: {answer}[/bold blue]")
            logger.info("[bold green]RAG chain executed successfully.[/bold green]")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error executing RAG chain: {e}")
            raise