from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging

logger = logging.getLogger("QueryAugmentationTool")

class QueryAugmentationTool():
    def generate_multiple_queries(llm, question: str, prompt: str):
        try:
            logger.info("Generating query perspectives.")
            prompt_perspectives = ChatPromptTemplate.from_template(prompt)

            perspective_chain = prompt_perspectives | llm | StrOutputParser()

            perspectives = perspective_chain.invoke({"question": question})
            
            queries = [q.strip() for q in perspectives.split("\n") if q.strip()]
            logger.info(f"Generated queries: {queries}")
            
            return queries
        except Exception as e:
            logger.error(f"Error generating queries: {e}")
            raise


