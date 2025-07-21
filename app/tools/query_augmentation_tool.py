from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.logger import logger

class MultiQueryTool():
    def generate_multiple_queries(llm, question: str, prompt: str):
        prompt_perspectives = ChatPromptTemplate.from_template(prompt)

        perspective_chain = prompt_perspectives | llm | StrOutputParser()

        logger.info("  |  Generating query perspectives...")
        perspectives = perspective_chain.invoke({"question": question})
        
        queries = [q.strip() for q in perspectives.split("\n") if q.strip()]
        logger.info("  |  Queries generated: {queries}")
        
        return queries
