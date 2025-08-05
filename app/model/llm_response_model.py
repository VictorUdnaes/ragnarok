from typing import Union, Any, Dict
from pydantic import BaseModel, Field
from enum import Enum
import uuid

class ResponseType(str, Enum):
    TEXT = "text"
    LIST = "list"
    DICT = "dict"
    CUSTOM = "custom"

class LLMResponse(BaseModel):
    step_id: str = str(uuid.uuid4())
    step_name: str
    response_type: ResponseType
    data: Any  # The actual pydantic object
    metadata: Dict[str, Any] = Field(default_factory=dict)