import time
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Chatbot API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    sessionId: str = "local-dev"
    message: str


class ChatResponse(BaseModel):
    messageId: str
    answer: str
    sources: list
    latencyMs: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    t0 = time.time()
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")

    answer = f"(session {req.sessionId}) You said: {msg}"

    return ChatResponse(
        messageId=str(uuid.uuid4()),
        answer=answer,
        sources=[],
        latencyMs=int((time.time() - t0) * 1000),
    )
