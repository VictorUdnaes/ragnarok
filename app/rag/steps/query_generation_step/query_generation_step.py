from rag.steps.abstract.abstract_step_preset import AbstractStepPreset
from logging import getLogger
from rag.steps.query_generation_step.queries_from_plan_preset import QueriesFromPlanPreset

logger = getLogger("QueryGenerationStep")

class QueryGenerationStep:
    def __init__(self):
        self.queries_from_plan_preset = QueriesFromPlanPreset()
