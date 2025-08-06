from model.anonymize_model import AnonymizedQuestion, DeanonymizedPlan
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from model.plan_model import Plan
from langchain.schema.runnable import RunnableSequence
from model.queries_from_plan import QueriesFromPlan
from prompts.prompt_manager import anonymizer_prompt, planner_prompt, deanonymize_prompt, queries_from_plan_prompt

class PlanningTool:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm


    
    

    def create_queries_from_plan(self, question: str, plan: Plan) -> QueriesFromPlan:
        return self \
            .build_chain(["question", "plan"], queries_from_plan_prompt, QueriesFromPlan) \
            .invoke({
                "question": question,
                "plan": plan.steps
            })

    
    def build_chain(self, input_variables: list[str], prompt: str, format_object) -> RunnableSequence:
        chain = (
            PromptTemplate(
                input_variables=input_variables,
                template=prompt
            ) | self.llm.with_structured_output(format_object)
        )

        return chain
    