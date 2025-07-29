from planning_tool import PlanningTool
from langchain_ollama import ChatOllama
from model.plan_model import Plan
from model.anonymize_model import AnonymizedQuestion

# Setup
llm = ChatOllama(model="llama3.1", temperature=0)
planner = PlanningTool(llm)
from planning_tool import PlanningTool
from app.openai_config import openapi_client

# Setup model variables and connection string
question = "Når går Venstre inn for å inkludere omskjæring i en lov mot omskjæring av kvinner?"

llm = openapi_client()

anonymized_question: AnonymizedQuestion = planner.anonymize_question(question)

if isinstance(anonymized_question, AnonymizedQuestion):
    # Create initial plan based on the anonymized question
    plan: Plan = planner.create_initial_plan(anonymized_question.anonymized_question)
    
    if not isinstance(plan, Plan):
        raise ValueError("Plan creation failed, expected a Plan instance")
    
    # Deanonymize the plan using the mapping from the anonymized question
    deanonymized_plan = planner.deanonymize_plan(plan.steps, anonymized_question.mapping)

    print(deanonymized_plan)



