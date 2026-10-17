import os
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# 导入所有的图构建函数
from agent import (
    build_chat_graph, build_react_graph, build_plan_act_graph, 
    build_rewoo_graph, build_codeact_graph, build_recode_graph
)

class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"

DB_PATH = "./checkpoints.db"
checkpointer = None
graphs = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global checkpointer, graphs
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as saver:
        await saver.setup()
        checkpointer = saver
        # 初始化 6 种模式的图
        graphs["chat"] = build_chat_graph(checkpointer)
        graphs["react"] = build_react_graph(checkpointer)
        graphs["plan_act"] = build_plan_act_graph(checkpointer)
        graphs["rewoo"] = build_rewoo_graph(checkpointer)
        graphs["codeact"] = build_codeact_graph(checkpointer)
        graphs["recode"] = build_recode_graph(checkpointer)
        yield

app = FastAPI(title="LangGraph Agent API", lifespan=lifespan, default_response_class=UTF8JSONResponse)

class ChatRequest(BaseModel):
    message: str
    thread_id: str
    mode: str = "chat"  # 可选 chat, react, plan_act, rewoo, codeact, recode

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    graph = graphs.get(req.mode)
    if not graph:
        raise HTTPException(status_code=400, detail="Unknown mode")

    async def event_generator():
        try:
            async for chunk, _meta in graph.astream(
                {"messages": [HumanMessage(content=req.message)]},
                config={"configurable": {"thread_id": req.thread_id}},
                stream_mode="messages",
            ):
                # 1. 思考过程 & 自定义事件
                reasoning = ""
                if hasattr(chunk, "additional_kwargs"):
                    reasoning = chunk.additional_kwargs.get("reasoning_content", "") or ""
                    code_block = chunk.additional_kwargs.get("code_block", "")
                    if code_block:
                        msg = f"💻 [生成的代码]\n```python\n{code_block}\n```\n\n"
                        payload = json.dumps({"type": "reasoning", "delta": msg}, ensure_ascii=False)
                        yield f"data: {payload}\n\n"

                if reasoning:
                    payload = json.dumps({"type": "reasoning", "delta": reasoning}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"

                # 2. 工具调用
                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        msg = f"\n🛠️ [调用工具] {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})\n"
                        payload = json.dumps({"type": "reasoning", "delta": msg}, ensure_ascii=False)
                        yield f"data: {payload}\n\n"

                # 3. 工具执行结果
                if getattr(chunk, "type", "") == "tool":
                    msg = f"👉 [工具返回] {chunk.content}\n\n"
                    payload = json.dumps({"type": "reasoning", "delta": msg}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"

                # 4. 最终回答文本
                content = getattr(chunk, "content", "") or ""
                if content:
                    payload = json.dumps({"type": "content", "delta": content}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            payload = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/history/{thread_id}")
async def history(thread_id: str):
    # 默认使用 chat 图去读取历史（因为 state 在同一个 DB 里）
    graph_instance = graphs.get("chat")
    if not graph_instance:
        raise HTTPException(status_code=500, detail="Graph not initialized")
        
    state = await graph_instance.aget_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    if not state.values or "messages" not in state.values:
        return {"thread_id": thread_id, "messages": []}

    messages = []
    for m in state.values["messages"]:
        item = {
            "role": m.__class__.__name__,
            "content": m.content,
            "reasoning_content": m.additional_kwargs.get("reasoning_content", "") if hasattr(m, "additional_kwargs") else ""
        }
        messages.append(item)

    return {"thread_id": thread_id, "messages": messages}

@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    if checkpointer is None:
        raise HTTPException(status_code=500, detail="Checkpointer not initialized")
    try:
        await checkpointer.adelete_thread(thread_id)
    except AttributeError:
        raise HTTPException(status_code=501, detail="当前 langgraph-checkpoint-sqlite 版本不支持 adelete_thread")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"deleted": thread_id}

@app.get("/health")
async def health():
    return {"status": "ok"}

app.mount("/", StaticFiles(directory="static", html=True), name="static")