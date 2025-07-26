from pydantic import BaseModel, Field

class RelevantContent(BaseModel):
    relevant_content: str = Field(description="The relevant content from the retrieved documents that is relevant to the query.")
