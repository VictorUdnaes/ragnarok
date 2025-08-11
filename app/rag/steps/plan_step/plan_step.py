from enum import Enum
from model.llm_response_model import LLMResponse, ResponseType
from typing import Dict, Any
from pydantic import BaseModel, Field
from tools.planning_tool import PlanningTool
from model.plan_model import Plan
from model.anonymize_model import AnonymizedQuestion, DeanonymizedPlan
from prompts.prompt_manager import anonymizer_prompt, planner_prompt, deanonymize_prompt
from langchain_openai import ChatOpenAI
import logging
from enum import Enum
from rag.steps.plan_step.anonymizer_preset import AnonymizerPreset

logger = logging.getLogger("ApplicationService")

# TODO: add functionality for not using anonymization
class PlanStep():
    def __init__(self):
        self.anonymizer_preset = AnonymizerPreset()
    