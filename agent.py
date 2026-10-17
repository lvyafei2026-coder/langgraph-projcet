import os
import json
import re
import traceback
import contextlib
import io
from typing import Annotated, TypedDict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph_codeact import create_codeact
from RestrictedPython import compile_restricted_exec

# ================= 1. 基础定义与工具 =================
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}今天晴转多云，气温 22-28 度。"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式。例如: '9.11 - 9.8'"""
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算失败: {str(e)}"

tools_list = [get_weather, calculate]

# ================= 2. LLM 初始化 =================
def get_llm(with_tools=False):
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0.7,
    )
    if with_tools:
        return llm.bind_tools(tools_list)
    return llm

# ================= 3. 普通 Chat 图 =================
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

# ================= 4. ReAct 图 =================
async def react_agent_node(state: MessagesState):
    llm = get_llm(with_tools=True)
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

def build_react_graph(checkpointer):
    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", react_agent_node)
    workflow.add_node("tools", ToolNode(tools_list))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition)
    workflow.add_edge("tools", "agent")
    return workflow.compile(checkpointer=checkpointer)

# ================= 5. Plan-Act 图 =================
class PlanActState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: list[str]
    current_step: int

async def planner_node(state: PlanActState):
    llm = get_llm()
    prompt = f"""你是任务规划专家。请将用户的请求拆解为 1-3 个步骤。
用户请求: {state['messages'][-1].content}
直接输出 JSON 格式的步骤列表，不要输出任何其他内容。例如: ["步骤1", "步骤2"]"""
    res = await llm.ainvoke([HumanMessage(content=prompt)])
    try:
        text = res.content.strip()
        if text.startswith("```json"): text = text[7:-3]
        plan = json.loads(text)
    except:
        plan = [f"直接回答问题: {state['messages'][-1].content}"]
    
    reasoning_msg = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": f"【📝 Plan-Act 规划阶段】\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"}
    )
    return {"plan": plan, "current_step": 0, "messages": [reasoning_msg]}

async def executor_node(state: PlanActState):
    llm = get_llm()
    step = state["plan"][state["current_step"]]
    context = "\n".join([f"- {m.content}" for m in state["messages"] if m.content])
    prompt = f"你正在执行任务计划。\n当前步骤: {step}\n已知信息: {context}\n请执行并给出结果。"
    res = await llm.ainvoke([HumanMessage(content=prompt)])
    
    reasoning_msg = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": f"【⚙️ Plan-Act 执行阶段 {state['current_step']+1}】\n执行: {step}\n结果: {res.content}\n\n"}
    )
    return {"current_step": state["current_step"] + 1, "messages": [reasoning_msg]}

async def finalizer_node(state: PlanActState):
    llm = get_llm()
    context = "\n".join([m.additional_kwargs.get("reasoning_content", "") for m in state["messages"] if hasattr(m, "additional_kwargs")])
    prompt = f"根据以下执行过程，给用户一个完整清晰的最终回复。\n执行过程:\n{context}"
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

# ================= 6. ReWOO 图 =================
class ReWOOState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: list[dict]
    evidences: str

async def rewoo_planner_node(state: ReWOOState):
    llm = get_llm()
    prompt = f"""你是任务规划专家。制定最多3个步骤的工具执行计划。
可用工具：calculate(expression: str), get_weather(city: str)
直接输出严格 JSON 格式: {{"plan": [{{"tool": "get_weather", "args": {{"city": "北京"}}}}]}}
如果不需要工具: {{"plan": []}}
用户问题: {state['messages'][-1].content}"""
    res = await llm.ainvoke([HumanMessage(content=prompt)])
    try:
        text = res.content.strip()
        if text.startswith("```json"): text = text[7:-3]
        if text.startswith("```"): text = text[3:-3]
        plan = json.loads(text).get("plan", [])
    except:
        plan = []
    reasoning_msg = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": f"【🧠 ReWOO 规划阶段】\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"}
    )
    return {"plan": plan, "messages": [reasoning_msg]}

async def rewoo_worker_node(state: ReWOOState):
    evidences = []
    for i, step in enumerate(state["plan"]):
        tool_name = step.get("tool")
        args = step.get("args", {})
        target_tool = next((t for t in tools_list if t.name == tool_name), None)
        if target_tool:
            try: result = target_tool.invoke(args)
            except Exception as e: result = f"失败: {e}"
        else:
            result = f"未找到工具 {tool_name}"
        evidences.append(f"#E{i+1}: 调用 {tool_name} 返回 -> {result}")
    evidence_str = "\n".join(evidences) if evidences else "无工具调用。"
    
    reasoning_msg = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": f"【⚙️ ReWOO 执行阶段】\n{evidence_str}\n\n"}
    )
    return {"evidences": evidence_str, "messages": [reasoning_msg]}

async def rewoo_solver_node(state: ReWOOState):
    llm = get_llm()
    prompt = f"根据已知信息回答用户问题。\n已知信息:\n{state.get('evidences', '无')}\n用户问题: {state['messages'][0].content}"
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

# ================= 7. CodeAct 图 =================
def secure_sandbox_eval(code: str, _locals: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    original_keys = set(_locals.keys())
    for t in tools_list:
        _locals[t.name] = t.func

    # 禁止 import，防止模块对象污染 State
    def safe_import(name, *args, **kwargs):
        raise ImportError(f"当前沙箱不允许使用 import {name}，请仅使用提供的工具或基础数学运算。")

    safe_builtins = {
        'print': print, 'len': len, 'range': range, 'str': str, 'int': int,
        'float': float, 'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
        'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError,
        '__import__': safe_import, # <-- 替换为安全的 import
        'abs': abs, 'sum': sum, 'min': min, 'max': max, 'round': round
    }
    
    try:
        byte_code = compile_restricted_exec(code)
        if byte_code.errors: return f"代码编译错误: {byte_code.errors}", _locals
        
        with contextlib.redirect_stdout(io.StringIO()) as f:
            exec(byte_code.code, {"__builtins__": safe_builtins}, _locals)
        result = f.getvalue() or "<代码执行成功，无输出>"
    except Exception as e:
        result = f"执行错误: {repr(e)}"
    
    # 过滤出新增的、且可以被 JSON 序列化的基础变量，防止 module / function 对象存入 State
    new_keys = set(_locals.keys()) - original_keys
    new_vars = {}
    for k in new_keys:
        if k.startswith('__') or k in [t.name for t in tools_list]:
            continue
        val = _locals[k]
        # 只保留基本数据类型
        if isinstance(val, (int, float, str, bool, list, dict, tuple, set, type(None))):
            try:
                json.dumps(val) # 二次验证是否可序列化
                new_vars[k] = val
            except:
                pass
                
    return result, new_vars

def build_codeact_graph(checkpointer):
    llm = get_llm()
    codeact_graph = create_codeact(
        model=llm,
        tools=tools_list,
        eval_fn=secure_sandbox_eval,
    )
    return codeact_graph.compile(checkpointer=checkpointer)

# ================= 8. ReCode 图 (递归代码生成) =================
class ReCodeState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    code: str
    execution_trace: str

async def recode_planner_node(state: ReCodeState):
    llm = get_llm()
    user_query = state['messages'][-1].content if state['messages'] else ""
    prompt = f"""你是代码规划专家。请为用户需求生成一个 Python 代码骨架。
要求：
1. 将高层计划写成抽象函数，函数体为 `raise NotImplementedError("待分解")`。
2. 在最后调用这些函数。
3. 只输出 Python 代码，不要用 markdown 包裹。

用户需求: {user_query}

示例输出：
def 查天气(city):
    raise NotImplementedError("待分解")

w1 = 查天气("北京")
print(w1)
"""
    res = await llm.ainvoke([HumanMessage(content=prompt)])
    code = res.content.strip()
    if code.startswith("```python"): code = code[9:-3]
    elif code.startswith("```"): code = code[3:-3]
    
    # 去掉内部可能引起冲突的 Markdown 反引号
    reasoning_msg = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": f"【📝 ReCode 规划阶段】\n{code}\n\n"}
    )
    return {"code": code, "execution_trace": "", "messages": [reasoning_msg]}

def _sandbox_execute_recode(code: str) -> str:
    def safe_import(name, *args, **kwargs):
        raise ImportError(f"当前沙箱不允许使用 import {name}。")

    safe_builtins = {
        'print': print, 'len': len, 'range': range, 'str': str, 'int': int,
        'float': float, 'list': list, 'dict': dict, 'Exception': Exception,
        'NotImplementedError': NotImplementedError, 'abs': abs, 'sum': sum,
        'min': min, 'max': max, 'round': round,
        '__import__': safe_import
    }
    _locals = {t.name: t.func for t in tools_list}
    
    try:
        with contextlib.redirect_stdout(io.StringIO()) as f:
            exec(code, {"__builtins__": safe_builtins}, _locals)
        output = f.getvalue().strip()
        return "__SUCCESS__" + (output if output else "（无输出）")
    except NotImplementedError as e:
        raise e
    except Exception as e:
        return f"__ERROR__执行错误: {repr(e)}"

def _extract_func_name_from_traceback() -> str:
    tb = traceback.format_exc()
    matches = re.findall(r'in (\w+)', tb)
    return matches[-1] if matches else "未知函数"

def _replace_function(full_code: str, func_name: str, new_func_code: str) -> str:
    pattern = rf"(def {func_name}\s*\(.*?\):)(.*?)(?=\ndef |\Z)"
    return re.sub(pattern, new_func_code, full_code, flags=re.DOTALL)

async def recode_executor_node(state: ReCodeState):
    llm = get_llm()
    code = state["code"]
    trace = state.get("execution_trace", "")
    max_steps = 5
    step = 0
    final_output = ""
    
    while step < max_steps:
        step += 1
        try:
            result = _sandbox_execute_recode(code)
            if result.startswith("__SUCCESS__"):
                final_output = result.replace("__SUCCESS__", "").strip()
                trace += f"\n✅ 第 {step} 次执行成功，最终输出:\n{final_output}\n"
                break
            else:
                trace += f"\n⚠️ 第 {step} 次执行代码报错:\n{result}\n"
                break
        except NotImplementedError:
            func_name = _extract_func_name_from_traceback()
            trace += f"\n🔍 第 {step} 次执行触发占位符: `{func_name}()`，开始递归分解...\n"
            
            # 去掉了 markdown 代码块标记，避免字符串转义冲突
            prompt = f"""你正在递归分解一个 Python 函数。
当前代码:
{code}

未实现的函数: `{func_name}`

请为 `{func_name}` 提供具体实现代码，直接输出该函数的完整定义，不要输出其他内容。
"""
            decomp_res = await llm.ainvoke([HumanMessage(content=prompt)])
            new_func_code = decomp_res.content.strip()
            if new_func_code.startswith("```python"): new_func_code = new_func_code[9:-3]
            elif new_func_code.startswith("```"): new_func_code = new_func_code[3:-3]
            
            code = _replace_function(code, func_name, new_func_code)
            trace += f"   ↳ 已生成实现:\n{new_func_code}\n"
        except Exception as e:
            trace += f"\n❌ 第 {step} 次执行发生未知错误: {repr(e)}\n"
            break
            
    reasoning_msg = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": f"【⚙️ ReCode 递归执行阶段】\n{trace}\n\n"}
    )
    
    content_msg = AIMessage(content=final_output if final_output else "执行完成，但无标准输出。请查看思考过程。")
    
    return {
        "code": code, 
        "execution_trace": trace,
        "messages": [reasoning_msg, content_msg]
    }

def build_recode_graph(checkpointer):
    workflow = StateGraph(ReCodeState)
    workflow.add_node("planner", recode_planner_node)
    workflow.add_node("executor", recode_executor_node)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", END)
    return workflow.compile(checkpointer=checkpointer)