from app.openai_config import openapi_client
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
llm = openapi_client()

class Response(BaseModel):
    answer: str

chain = (
    PromptTemplate(
        input_variables=[],
        template="Give me a detailed comparison of GraphQL versus GRPC, including their advantages and disadvantages. Provide examples of use cases for each technology."
    ) | llm
)

response = chain.invoke({})
print("")
print(f"Response: {response.content}")