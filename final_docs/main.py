from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Optional, List, Dict
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_agent
from langgraph.types import Command
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.managed import RemainingSteps
from typing_extensions import Annotated, NotRequired, TypedDict
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import HumanMessage, AIMessage
from typing import Literal, List, Any
from langchain_core.tools import tool
from langgraph.types import Command
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')


members_dict = {'information_node':'specialized agent to provide information related to availability of doctors or any FAQs related to hospital.','booking_node':'specialized agent to only to book, cancel or reschedule appointment'}

options = list(members_dict.keys()) + ["FINISH"]

worker_info = '\n\n'.join([f'WORKER: {member} \nDESCRIPTION: {description}' for member, description in members_dict.items()]) + '\n\nWORKER: FINISH \nDESCRIPTION: If User Query is answered and route to Finished'

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the following workers. "
    "### SPECIALIZED ASSISTANT:\n"
    f"{worker_info}\n\n"
    "Your primary role is to help the user make an appointment with the doctor and provide updates on FAQs and doctor's availability. "
    "If a customer requests to know the availability of a doctor or to book, reschedule, or cancel an appointment, "
    "delegate the task to the appropriate specialized workers. Each worker will perform a task and respond with their results and status. "
    "When all tasks are completed and the user query is resolved, respond with FINISH.\n\n"

    "**IMPORTANT RULES:**\n"
    "1. If the user's query is clearly answered and no further action is needed, respond with FINISH.\n"
    "2. If you detect repeated or circular conversations, or no useful progress after multiple turns, return FINISH.\n"
    "3. If more than 10 total steps have occurred in this session, immediately respond with FINISH to prevent infinite recursion.\n"
    "4. Always use previous context and results to determine if the user's intent has been satisfied. If it has — FINISH.\n"
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    ui_type: Optional[str] = None
    text: Optional[str] = None
    data: Optional[dict] = None
    hello_world: Optional[str] = "This is the hello world default value"
 
class Specialty(BaseModel):
    specialty_id: int = Field(..., description="Unique ID of the specialty")
    name: str = Field(..., description="Specialty name e.g. Pediatrics")
    description_val: str = Field(..., description="Description of the specialty")

class SymptomSchema(BaseModel):
    is_symptom: bool = Field(..., description="Whether a symptom was detected or not")
    symptom: str = Field(..., description="User’s reported symptom(s)")
    specialties: List[Specialty] = Field(..., description="List of most relevant medical specialties for the symptom")

structured_model = llm.with_structured_output(SymptomSchema)

class Router(TypedDict):
    next: Literal["information_node", "booking_node", "FINISH"]
    reasoning: str


def supervisor_node(self, state: AgentState) -> Command[Literal['information_node', 'booking_node', '__end__']]:
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user's identification number is {state['id_number']}"},
        ] + state["messages"]
    
    query = ''
    if len(state['messages']) == 1:
        query = state['messages'][0].content
    
    response = self.llm_model.with_structured_output(Router).invoke(messages)
    
    goto = response["next"]
    if goto == "FINISH":
        goto = END
        
    if query:
        return Command(goto=goto, update={'next': goto, 
                                        'query': query, 
                                        'current_reasoning': response["reasoning"],
                                        'messages': [HumanMessage(content=f"user's identification number is {state['id_number']}")]
                        })
    return Command(goto=goto, update={'next': goto, 
                                        'current_reasoning': response["reasoning"]}
                    )

def information_node(self, state: AgentState) -> Command[Literal['supervisor']]:
    system_prompt = "You are specialized agent to provide information related to availability of doctors or any FAQs related to hospital based on the query. You have access to the tool.\n Make sure to ask user politely if you need any further information to execute the tool.\n For your information, Always consider current year is 2024."
    
    system_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt
                ),
                (
                    "placeholder", 
                    "{messages}"
                ),
            ]
        )
    
    information_agent = create_react_agent(model=self.llm_model,tools=[check_availability_by_doctor,check_availability_by_specialization] ,prompt=system_prompt)
    
    result = information_agent.invoke(state)
    
    return Command(
        update={
            "messages": state["messages"] + [
                AIMessage(content=result["messages"][-1].content, name="information_node")
                #HumanMessage(content=result["messages"][-1].content, name="information_node")
            ]
        },
        goto="supervisor",
    )

def booking_node(self, state: AgentState) -> Command[Literal['supervisor']]:
    print("*****************called booking node************")

    system_prompt = "You are specialized agent to set, cancel or reschedule appointment based on the query. You have access to the tool.\n Make sure to ask user politely if you need any further information to execute the tool.\n For your information, Always consider current year is 2024."
    
    system_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt
                ),
                (
                    "placeholder", 
                    "{messages}"
                ),
            ]
        )
    booking_agent = create_react_agent(model=self.llm_model,tools=[set_appointment,cancel_appointment,reschedule_appointment],prompt=system_prompt)

    result = booking_agent.invoke(state)
    
    return Command(
        update={
            "messages": state["messages"] + [
                AIMessage(content=result["messages"][-1].content, name="booking_node")
            ]
        },
        goto="supervisor",
    )

 
graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("information_node", information_node)
graph.add_node("booking_node", booking_node)
graph.add_edge(START, "supervisor")
app = graph.compile()
