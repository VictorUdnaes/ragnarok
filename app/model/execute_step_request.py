from typing import Any, Dict, Type
from pydantic import BaseModel

class ExecuteStepRequest(BaseModel):
    """
    Request model for executing a single RAG step.

    Params:
        step_name, preset, spec, context
    """
    step_name: str
    preset: str
    spec: Dict[str, Any]  # Raw dict
    context: Dict[str, Any] = {}
    
    def deserialize(self, spec_class: Type[BaseModel]) -> BaseModel:
        """Deserialize spec dict into the correct spec class"""
        return spec_class(**self.spec)