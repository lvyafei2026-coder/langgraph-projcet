import os
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse          # ← 加 StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import build_graph


class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"


DB_PATH = "./checkpoints.db"

checkpointer = None
graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global checkpointer, graph
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as saver:
        await saver.setup()
        checkpointer = saver
        graph = build_graph(checkpointer)
        yield


app = FastAPI(
    title="LangGraph Agent API",
    lifespan=lifespan,
    default_response_class=UTF8JSONResponse,
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str


class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    message_count: int


# ---------- 原接口：一次性返回 ----------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if graph is None:
        raise HTTPException(status_code=500, detail="Agent graph not initialized")
    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=req.message)]},
            config={"configurable": {"thread_id": req.thread_id}},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    last = result["messages"][-1]
    return ChatResponse(
        thread_id=req.thread_id,
        reply=last.content,
        message_count=len(result["messages"]),
    )


# ---------- 新增：SSE 流式输出 ----------
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if graph is None:
        raise HTTPException(status_code=500, detail="Agent graph not initialized")

    async def event_generator():
        try:
            async for chunk, _meta in graph.astream(
                {"messages": [HumanMessage(content=req.message)]},
                config={"configurable": {"thread_id": req.thread_id}},
                stream_mode="messages",
            ):
                content = getattr(chunk, "content", "") or ""
                if content:
                    payload = json.dumps({"delta": content}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            payload = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",          # 防止中间层缓冲（Render 上很关键）
        },
    )


# ---------- 原接口：历史 ----------
@app.get("/history/{thread_id}")
async def history(thread_id: str):
    if graph is None:
        raise HTTPException(status_code=500, detail="Agent graph not initialized")
    state = await graph.aget_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    if not state.values or "messages" not in state.values:
        return {"thread_id": thread_id, "messages": []}

    return {
        "thread_id": thread_id,
        "messages": [
            {"role": m.__class__.__name__, "content": m.content}
            for m in state.values["messages"]
        ],
    }


# ---------- 新增：删除会话 ----------
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


# ---------- 静态资源（必须最后挂载） ----------
app.mount("/", StaticFiles(directory="static", html=True), name="static")