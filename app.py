import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import build_graph

# ---------- 数据库路径 ----------
# 注意：Render 免费层没有持久化磁盘，容器重启后该文件会丢失，短期记忆会重置。
# 如果需要持久化，可以升级 Render 磁盘，或把状态存到外部数据库（如 Neon Postgres）。
DB_PATH = "./checkpoints.db"

checkpointer = None
graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global checkpointer, graph
    
    # 使用 async with 管理 AsyncSqliteSaver 的生命周期
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as saver:
        await saver.setup()
        checkpointer = saver
        graph = build_graph(checkpointer)
        yield
    # 退出时连接会自动关闭

app = FastAPI(title="LangGraph Agent API", lifespan=lifespan)

# ---------- 请求/响应模型 ----------
class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ChatResponse(BaseModel):
    thread_id: str
    reply: str
    message_count: int

# ---------- 路由 ----------
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