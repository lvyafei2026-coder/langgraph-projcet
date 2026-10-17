import os
import json
import operator
from typing import Annotated, TypedDict, Union

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ================= 1. 定义工具 =================
@tool
def calculate(expression: str) -> str:
    """计算数学表达式。例如: '9.11 - 9.8', '2 * 3 + 4'"""
    try:
        # 仅做演示，实际生产环境请勿直接用 eval
        result = eval(expression, {"__builtins__": None}, {})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算失败: {str(e)}"

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气。"""
    # 模拟天气 API
    return f"{city}今天晴转多云，气温 22-28 度，适合出行。"

tools_list = [calculate, get_weather]

# ================= 2. LLM 初始化 =================
def get_llm(with_tools=False):
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-reasoner"), # 建议默认用 reasoner
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0.7,
    )
    if with_tools:
        return llm.bind_tools(tools_list)
    return llm

# ================= 3. 构建 Chat 图 =================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def call_model(state: ChatState):
    llm = get_llm()
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

def build_chat_graph(checkpointer):
    workflow = StateGraph(ChatState)
    workflow.add_node("agent", call_model)
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)
    return workflow.compile(checkpointer=checkpointer)

# ================= 4. 构建 ReAct 图 =================
# ReAct: 思考 -> 调用工具 -> 观察结果 -> 再次思考，直到输出最终答案
async def react_agent_node(state: MessagesState):
    llm = get_llm(with_tools=True)
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

def build_react_graph(checkpointer):
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", react_agent_node)
    workflow.add_node("tools", ToolNode(tools_list))
    
    workflow.add_edge(START, "agent")
    # 如果模型决定调用工具，走向 tools 节点；否则结束
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile(checkpointer=checkpointer)

# ================= 5. 构建 Plan-Act 图 =================
# Plan-Act: 先生成计划 (Plan)，再逐步执行 (Act)
class PlanActState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: list[str]
    current_step: int

async def planner_node(state: PlanActState):
    llm = get_llm()
    prompt = f"""你是任务规划专家。请将用户的请求拆解为 1-3 个步骤。
用户请求: {state['messages'][-1].content}

请直接输出 JSON 格式的步骤列表，不要输出任何其他内容。
例如: ["步骤1内容", "步骤2内容"]
"""
    res = await llm.ainvoke([HumanMessage(content=prompt)])
    # 简单解析 JSON
    try:
        text = res.content.strip()
        if text.startswith("```json"): text = text[7:-3]
        plan = json.loads(text)
    except:
        plan = [f"直接回答问题: {state['messages'][-1].content}"]
    
    # 将规划过程包装成 reasoning_content 供前端显示
    reasoning_msg = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": f"【规划阶段】\n制定计划:\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"}
    )
    return {"plan": plan, "current_step": 0, "messages": [reasoning_msg]}

async def executor_node(state: PlanActState):
    llm = get_llm()
    step = state["plan"][state["current_step"]]
    # 将之前的执行结果作为上下文
    context = "\n".join([f"- {m.content}" for m in state["messages"] if m.content])
    
    prompt = f"""你正在执行任务计划。
当前步骤: {step}
已知信息: {context}
请执行该步骤并给出结果。"""
    
    res = await llm.ainvoke([HumanMessage(content=prompt)])
    
    reasoning_msg = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": f"【执行阶段 步骤{state['current_step']+1}】\n执行: {step}\n结果: {res.content}\n\n"}
    )
    return {"current_step": state["current_step"] + 1, "messages": [reasoning_msg]}

async def finalizer_node(state: PlanActState):
    llm = get_llm()
    context = "\n".join([m.additional_kwargs.get("reasoning_content", "") for m in state["messages"] if hasattr(m, "additional_kwargs")])
    prompt = f"""根据以下计划执行过程，给用户一个完整、清晰的最终回复。
执行过程:
{context}
"""
    res = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"messages": [res]}

def should_continue_plan(state: PlanActState):
    if state["current_step"] < len(state["plan"]):
        return "executor"
    return "finalizer"

def build_plan_act_graph(checkpointer):
    workflow = StateGraph(PlanActState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("finalizer", finalizer_node)
    
    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges("planner", should_continue_plan)
    workflow.add_conditional_edges("executor", should_continue_plan)
    workflow.add_edge("finalizer", END)
    
    return workflow.compile(checkpointer=checkpointer)

# ================= 6. 构建 ReWOO 图 =================
# ReWOO: 规划器生成所有工具调用计划 -> 执行器依次执行 -> 解决器汇总生成回答

class ReWOOState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: list[dict]     # 规划步骤，例如 [{"tool": "get_weather", "args": {"city": "北京"}}]
    evidences: str       # 工具执行结果集合

async def rewoo_planner_node(state: ReWOOState):
    llm = get_llm()
    prompt = f"""你是任务规划专家。请根据用户的问题，制定一个最多包含3个步骤的执行计划。
可用工具：
- calculate(expression: str) # 数学计算
- get_weather(city: str)     # 查天气

请直接输出严格的 JSON 格式，不要输出其他内容。格式如下：
{{
  "plan": [
    {{"tool": "get_weather", "args": {{"city": "北京"}}}},
    {{"tool": "calculate", "args": {{"expression": "22 + 5"}}}}
  ]
}}
如果不需要使用工具，请返回: {{"plan": []}}

用户问题: {state['messages'][-1].content}
"""
    res = await llm.ainvoke([HumanMessage(content=prompt)])
    
    # 解析 JSON
    try:
        text = res.content.strip()
        if text.startswith("```json"): text = text[7:-3]
        if text.startswith("```"): text = text[3:-3]
        data = json.loads(text)
        plan = data.get("plan", [])
    except:
        plan = []
    
    reasoning_msg = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": f"【🧠 ReWOO 规划阶段】\n生成的计划:\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"}
    )
    return {"plan": plan, "messages": [reasoning_msg]}

async def rewoo_worker_node(state: ReWOOState):
    # 依次执行计划中的工具
    evidences = []
    for i, step in enumerate(state["plan"]):
        tool_name = step.get("tool")
        args = step.get("args", {})
        
        # 找到对应的工具
        target_tool = next((t for t in tools_list if t.name == tool_name), None)
        if target_tool:
            try:
                result = target_tool.invoke(args)
            except Exception as e:
                result = f"执行失败: {str(e)}"
        else:
            result = f"未找到工具 {tool_name}"
            
        evidences.append(f"#E{i+1}: 调用 {tool_name}({json.dumps(args, ensure_ascii=False)}) 返回 -> {result}")

    evidence_str = "\n".join(evidences) if evidences else "无工具调用。"
    
    reasoning_msg = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": f"【⚙️ ReWOO 执行阶段】\n{evidence_str}\n\n"}
    )
    return {"evidences": evidence_str, "messages": [reasoning_msg]}

async def rewoo_solver_node(state: ReWOOState):
    llm = get_llm()
    prompt = f"""你是一个聪明的助手。请根据已知的工具执行结果，回答用户的问题。

已知信息:
{state.get('evidences', '无')}

用户问题: {state['messages'][0].content}
"""
    res = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"messages": [res]}

def build_rewoo_graph(checkpointer):
    workflow = StateGraph(ReWOOState)
    workflow.add_node("planner", rewoo_planner_node)
    workflow.add_node("worker", rewoo_worker_node)
    workflow.add_node("solver", rewoo_solver_node)
    
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "worker")
    workflow.add_edge("worker", "solver")
    workflow.add_edge("solver", END)
    
    return workflow.compile(checkpointer=checkpointer)