import os
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def get_llm():
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),          # ← 改为 DeepSeek 模型
        api_key=os.getenv("DEEPSEEK_API_KEY"),                    # ← 改为 DeepSeek Key
        base_url="https://api.deepseek.com/v1",                   # ← 指向 DeepSeek
        temperature=0.7,
    )

async def call_model(state: AgentState) -> dict:
    llm = get_llm()
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

def build_graph(checkpointer):
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    return workflow.compile(checkpointer=checkpointer)