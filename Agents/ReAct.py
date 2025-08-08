from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from langchain_google_genai import ChatGoogleGenerativeAI

GEMINI_API_KEY="AIzaSyC63pAqx5CeN4L585aACT2z3w1IGhd_OCY"

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]   #  means: "This is a list of BaseMessages, and when we update it, follow the special rule called add_messages.(concate not overide)"

# Example: Annotated[int, "some info"] means "this is an int, with some extra info".

# Sequence is a generic type from typing that means "any ordered collection" (like a list or tuple).

# BaseMessage is a custom class or type (probably defined somewhere else in your code).

# So, Sequence[BaseMessage] means "a list or tuple of BaseMessage objects".



############################  Imortant question,,,,,,,How the llm decides that it would need to call that specific tooll
# LangChain (and LangGraph) lets the LLM see a list of tools and their descriptions.
# Then during conversation, the LLM uses its own internal reasoning (from training) to decide whether to call a tool based on:

# The input message

# Tool names and docstrings

# System prompt context

# The LLM's own trained reasoning abilities


# model = ChatOpenAI(model = "gpt-4o").bind_tools(tools)

# This tells the model:

# “Hey, these are the tools you are allowed to use. Here's what they do (from their docstrings). Use them only if you think they help.”




# {
#   "tool_calls": [
#     {"tool_name": "add", "args": {"a": 40, "b": 12}},
#     {"tool_name": "multiply", "args": {"a": 52, "b": 6}}
#   ]
# }


# Summary in One Line:
# The LLM uses the user message + tool descriptions + system prompt + its own reasoning to decide if and which tool to call.

##############################################################







@tool
def add(a: int, b:int):
    """This is an addition function that adds 2 numbers together"""

    return a + b 

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

tools = [add, subtract, multiply]

# model = ChatOpenAI(model = "gpt-4o").bind_tools(tools)

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=GEMINI_API_KEY).bind_tools(tools)


def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}                  ########## another way of writing updated state,,,,,,,add_messages   import handles it how.......


def should_continue(state: AgentState): 
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)


tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):

    print("streaming" )

    print(stream)
    for s in stream:
        print(".........................")
        print(s)
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]}
print_stream(app.stream(inputs, stream_mode="values"))



#################two things   how pretty print#################33

 #   messages: Annotated[Sequence[BaseMessage], add_messages]   #  means: "This is a list of BaseMessages, and when we update it, follow the special rule called add_messages.(concate not overide)"
