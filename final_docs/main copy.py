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

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini')
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.managed import RemainingSteps
from typing_extensions import Annotated, NotRequired, TypedDict

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
    

class ExtractionOutput(BaseModel):
    messages: List[BaseMessage]   
    ui_type: Optional[str] = None
    text: Optional[str] = None
    data: Optional[Dict] = None
    hello_world: Optional[str] = "This is the hello world default value"


# @tool("check_symptom_speciality_tool")
# def check_symptom_speciality_tool(query, tool_call_id: Annotated[str, InjectedToolCallId]) -> ExtractionOutput:
#     """
#     Fetches symptoms and speciality information and returns it as a combined command.

#     Parameters:
#         query (str): The current query of the user.
         

#     Returns:
#         ExtractionOutput: response and dict object.
#     """

    
        
#     specialties = [
#         {"name": "General Physician", "description": "Specializes in general physician treatments and care.", "specialty_id": "97"},
#         {"name": "Aerospace Medicine", "description": "Specializes in aerospace medicine treatments and care.", "specialty_id": "11"},
#         {"name": "Cardiothoracic", "description": "Specializes in cardiothoracic treatments and care.", "specialty_id": "2"},
#         {"name": "Cardiothoracic Surgery", "description": "Specializes in cardiothoracic surgery treatments and care.", "specialty_id": "1"},
#         {"name": "Orthopedic", "description": "Specializes in orthopedic treatments and care.", "specialty_id": "3"}
#     ]
#     prompt = f"Extract all explicit and implicit health symptoms reported in user query: '{query}'. Use the provided specialties list to identify and return exactly two most relevant specialties.: {specialties}."

#     response = structured_model.invoke(prompt)
#     state_update = {}
#     if response.is_symptom:
#         state_update = {
#             "ui_type": "specialty_selection",
#             "text": "Based on your symptoms, here are the recommended specialties. Click any card to continue:",
#             "data": {
#                 "title": "Recommended Specialties",
#                 "subtitle": "Select a specialty to see available doctors",
#                 "specialties": [i.model_dump() for i in response.specialties]
#             }
#         }  

    
#     # import pdb; pdb.set_trace()
#     # print("Tool Response:", response)
#     # return Command(update=state_update)
#     state_update["hello_world"] = "This is updated hello world value from tool"
#     import pdb; pdb.set_trace()
#     return state_update


@tool(
    "check_symptom_speciality_tool",
    description="check_symptom_speciality_tool: Use this tool when user shows any symptom or asks for appointment.",
)
def check_symptom_speciality_tool(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    return Command(update={
        "example_state_key": "This is an extra value",
        "messages": [
            ToolMessage(
                content="I'm share the list of the specialty please select one of them for booking appointment.",             
                tool_call_id=tool_call_id
            )
        ]
    })

make_appointment_agent = create_agent(
    model=llm,
    tools=[check_symptom_speciality_tool],
    system_prompt="""

    # [ROLE]: You are a health assistant. Use the provided tools to ground your answers
    in health issue or a request. Be concise and factual.

    ## Tools:
    - `check_symptom_speciality_tool`: Use this when user shows any symptom or asks for appointment.

    ## Guidelines:
    - Be decisive: when you have sufficient information to act, proceed with tool calls without
    asking for confirmation. Only if information is missing or uncertain, ask a concise 
    clarifying question.
    - When preparing or describing actions, include appropriate parameters (e.g., state) based on available data. Do not fabricate numbers or facts.

    """,
    name="make_appointment_agent"
)




supervisor = create_supervisor(
    agents=[make_appointment_agent],
    model=ChatOpenAI(model="gpt-4o-mini"),
    output_mode="full_history",
    prompt="""
You are a Supervisor Agent responsible for routing user requests to the correct specialized agent.

- make_appointment_agent: Handles messages where user describes symptoms, feels unwell, or asks to book an appointment.

Your goal is to satisfy the user’s intent with minimal steps.


Routing:
- If the request is about health or booking appointment or symptoms, ask make_appointment_agent.

Handoff:
- From make_appointment_agent expect: one symptom or a bookappointment request.

Output:
- Donot explain your reasoning to the user, only return short consise responses.
- Always response in short concise manner.
    """,
    name="supervisor_agent",
    # response_format=ExtractionOutput
)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

def print_messages(messages, truncate_length=200):
    """
    Print messages with truncation for long tool message content.
    
    Args:
        messages: List of LangChain messages to print
        truncate_length: Maximum length before truncating tool message content
    """
    for message in messages:
        if isinstance(message, ToolMessage):
            print(f"=================================[1m Tool Message [0m=================================")
            print(f"Name: {message.name}\n")
            
            # Truncate long content
            content = message.content
            if len(content) > truncate_length:
                print(f"{content[:truncate_length]}...\n[Content truncated - {len(content)} chars total]")
            else:
                print(content)
        else:
            # Use pretty_print for AI and Human messages
            message.pretty_print()





def pretty_print_messages(messages):
    print("---")
    for message in messages:
        if hasattr(message, "name") and message.name:
            print(f"[{message.name.upper()}]: {message.content}")
        else:
            print(f"[{message.type.upper()}]: {message.content}")
    print("---")

# while True:
#     try:
#         query = input("Enter your query: ")
#         if query == 'q':
#             break
#         initial_state = {"messages": [HumanMessage(content=query)], "ui_type": None, "text": None, "data": None}
#         response = workflow.invoke(initial_state, config={"configurable": {"thread_id": "1"}})
#         print("Final State Result:", response.get("state", "No state found in response"))
#         for message in response["messages"]:
#             pretty_print_messages([message])
#     except Exception as e:
#         print(e)
#         import traceback
#         traceback.print_exc()
import uuid
config = {"configurable": {"thread_id":str(uuid.uuid4())}}
initial_state = {
  "messages": [HumanMessage(content="I have fever and backpain from lastnigth")], 
  "ui_type": None, 
  "text": None, 
  "data": None, 
  "hello_world": "test"
}
checkpointer = InMemorySaver()
workflow = supervisor.compile(checkpointer=checkpointer)

# graph = StateGraph(ChatState)
# graph.add_node("supervisor", supervisor)
# graph.add_node("make_appointment_agent", make_appointment_agent)

# graph.add_edge(START, "supervisor")
# graph.add_edge("make_appointment_agent", "supervisor")
 
# workflow = graph.compile(checkpointer=checkpointer)
# response = workflow.invoke(initial_state, config=config)


# supervisor = (
#     StateGraph(ChatState)
#     # NOTE: `destinations` is only needed for visualization and doesn't affect runtime behavior
#     .add_node(supervisor, destinations=("make_appointment_agent",  END))
#     .add_node(make_appointment_agent)
#     .add_edge(START, "supervisor")
#     # always return back to the supervisor
#     .add_edge("make_appointment_agent", "supervisor")
#     .compile()
# )
# response = workflow.invoke(initial_state, config=config)
# print("Final State Result ui_type:", response.get("ui_type", "No state found in response"))

# for message in response["messages"]:
#     pretty_print_messages([message])

# ai_output = response.get("messages")[-1].content
# print("AI Output:", ai_output)

# # 
# from langgraph.graph import StateGraph, START

# # 1. Initialize the App Graph with your custom state
# app_graph = StateGraph(ChatState)

# # 2. Add the supervisor as a node. The supervisor is a Runnable/Graph.
# app_graph.add_node("supervisor", supervisor)

# # 3. Define the simple flow: Start -> Supervisor (handles routing) -> End
# app_graph.set_entry_point(START)
# # Use the state of the supervisor (which will be an agent's response) 
# # to transition back to the supervisor or END.
# # In this simplified setup, we'll go straight to END for a single turn.
# app_graph.add_edge(START, "supervisor")
# app_graph.add_edge("supervisor", END)

# # 4. Compile the Final Application
# app = app_graph.compile()


 

initial_state = {
    "messages": [HumanMessage(content="I have fever and backpain from last night")], 
    "ui_type": None, 
    "text": None, 
    "data": None, 
    "hello_world": "test"
}
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

response = workflow.invoke(initial_state, config=config)
print("Final State Result ui_type:", response.get("ui_type", "No state found"))
for message in response["messages"]:
    pretty_print_messages([message])

print(response)

# print_messages(response["messages"])


"""
https://docs.langchain.com/oss/python/langchain/supervisor?utm_source=chatgpt.com 
https://docs.langchain.com/oss/python/langchain/multi-agent
https://docs.langchain.com/oss/python/langchain/human-in-the-loop
"""