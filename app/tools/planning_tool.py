from model.anonymize_model import AnonymizedQuestion, DeanonymizedPlan
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from utils.prompts import anonymizer_prompt, deanonymize_prompt, planner_prompt
from model.plan_model import Plan
from langchain.schema.runnable import RunnableSequence

class PlanningTool:
    def __init__(self, llm: ChatOllama):
        self.llm = llm

    def anonymize_question(self, question: str) -> AnonymizedQuestion:
        return self \
            .build_chain(["question"], anonymizer_prompt, AnonymizedQuestion) \
            .invoke(input=question)
    
    def create_initial_plan(self, question: str) -> Plan:
        return self \
            .build_chain(["question"], planner_prompt, Plan) \
            .invoke(input=question)

    def deanonymize_plan(self, plan: str, mapping:str) -> DeanonymizedPlan:
        return self \
            .build_chain(["plan", "mapping"], deanonymize_prompt, DeanonymizedPlan) \
            .invoke({
                "plan": plan,
                "mapping": mapping
            })
    
    def build_chain(self, input_variables: list[str], prompt: str, format_object) -> RunnableSequence:
        chain = (
            PromptTemplate(
                input_variables=input_variables,
                template=prompt
            ) | self.llm.with_structured_output(format_object)
        )

        return chain
    