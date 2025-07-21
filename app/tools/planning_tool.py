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
        anonymize_question_chain = (
            PromptTemplate(
                input_variables=["question"],
                partial_variables={"format_instructions": JsonOutputParser(pydantic_object=AnonymizedQuestion).get_format_instructions()},
                template=anonymizer_prompt
            )
        ) | self.llm.with_structured_output(AnonymizedQuestion)

        return anonymize_question_chain.invoke(question)
    
    def create_initial_plan(self, question: str) -> Plan:
        initial_plan_chain = PromptTemplate(
            template=planner_prompt,
            input_variables=["question"], 
        ) | self.llm.with_structured_output(Plan)

        return initial_plan_chain.invoke(input=question)

    def deanonymize_plan(self, plan: str, mapping:str) -> DeanonymizedPlan:
        deanonymize_plan_chain = (
            PromptTemplate(
                input_variables=["plan", "mapping"],
                template=deanonymize_prompt
            )
        ) | self.llm.with_structured_output(DeanonymizedPlan)

        return deanonymize_plan_chain.invoke({
            "plan": plan,
            "mapping": mapping
        })
    