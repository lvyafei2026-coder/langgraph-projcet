import os
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
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


# ---------- 流式输出：区分 reasoning_content 和 content ----------
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
                # ---- 提取思考过程 ----
                reasoning = ""
                if hasattr(chunk, "additional_kwargs"):
                    reasoning = chunk.additional_kwargs.get("reasoning_content", "") or ""

                if reasoning:
                    payload = json.dumps({"type": "reasoning", "delta": reasoning}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"

                # ---- 提取最终回答 ----
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
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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