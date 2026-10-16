import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles                 # ← 新增
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import build_graph


# ---------- 强制 UTF-8 响应 ----------
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


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------- 静态文件挂载（必须放最后） ----------
app.mount("/", StaticFiles(directory="static", html=True), name="static")