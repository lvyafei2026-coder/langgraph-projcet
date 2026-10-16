import os
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ---------- 状态定义 ----------
class AgentState(TypedDict):
    # 使用 add_messages 会自动将新消息追加到列表中，实现短期记忆的基础
    messages: Annotated[list[BaseMessage], add_messages]

# ---------- LLM ----------
def get_llm():
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
    )

# ---------- 节点 ----------
async def call_model(state: AgentState) -> dict:
    llm = get_llm()
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

# ---------- 构建图 ----------
def build_graph(checkpointer):
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    return workflow.compile(checkpointer=checkpointer)