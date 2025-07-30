from pydantic import BaseModel, Field

class RelevantContent(BaseModel):
    relevant_content_as_string: str = Field(description="The content from the retrieved documents that is relevant to the query.")
