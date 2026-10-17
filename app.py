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
from agent import build_chat_graph, build_react_graph, build_plan_act_graph

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
        # 初始化三种模式的图
        graphs["chat"] = build_chat_graph(checkpointer)
        graphs["react"] = build_react_graph(checkpointer)
        graphs["plan_act"] = build_plan_act_graph(checkpointer)
        yield

app = FastAPI(title="LangGraph Agent API", lifespan=lifespan, default_response_class=UTF8JSONResponse)

class ChatRequest(BaseModel):
    message: str
    thread_id: str
    mode: str = "chat"  # 默认 chat，可选 react, plan_act

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    graph = graphs.get(req.mode)
    if not graph:
        raise HTTPException(status_code=400, detail="Unknown mode")

    async def event_generator():
        try:
            # 统一使用 stream_mode="messages"
            async for chunk, _meta in graph.astream(
                {"messages": [HumanMessage(content=req.message)]},
                config={"configurable": {"thread_id": req.thread_id}},
                stream_mode="messages",
            ):
                # 1. 提取思考过程 (DeepSeek 原生或者 Plan-Act 模拟的)
                reasoning = ""
                if hasattr(chunk, "additional_kwargs"):
                    reasoning = chunk.additional_kwargs.get("reasoning_content", "") or ""

                if reasoning:
                    payload = json.dumps({"type": "reasoning", "delta": reasoning}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"

                # 2. 拦截工具调用事件 (ReAct 模式特有)
                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    for tc in chunk.tool_calls:
                        msg = f"\n🛠️ [调用工具] {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})\n"
                        payload = json.dumps({"type": "reasoning", "delta": msg}, ensure_ascii=False)
                        yield f"data: {payload}\n\n"

                # 3. 拦截工具执行结果 (ToolMessage)
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

# ---------- 历史查询：同时返回 reasoning_content ----------
@app.get("/history/{thread_id}")
async def history(thread_id: str):
    if graph is None:
        raise HTTPException(status_code=500, detail="Agent graph not initialized")
    state = await graph.aget_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    if not state.values or "messages" not in state.values:
        return {"thread_id": thread_id, "messages": []}

    messages = []
    for m in state.values["messages"]:
        item = {
            "role": m.__class__.__name__,
            "content": m.content,
        }
        # 从 additional_kwargs 中提取思考过程
        if hasattr(m, "additional_kwargs"):
            item["reasoning_content"] = m.additional_kwargs.get("reasoning_content", "") or ""
        else:
            item["reasoning_content"] = ""
        messages.append(item)

    return {"thread_id": thread_id, "messages": messages}


# ---------- 删除会话 ----------
@app.delete("/threads/{thread_id}")
async def delete_thread(thread_id: str):
    if checkpointer is None:
        raise HTTPException(status_code=500, detail="Checkpointer not initialized")
    try:
        await checkpointer.adelete_thread(thread_id)
    except AttributeError:
        raise HTTPException(
            status_code=501,
            detail="当前 langgraph-checkpoint-sqlite 版本不支持 adelete_thread",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"deleted": thread_id}


@app.get("/health")
async def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")