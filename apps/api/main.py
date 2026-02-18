import time
import uuid
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx
from dotenv import load_dotenv
from db import init_pool, close_pool, get_pool
from chunking import chunk_text
from embeddings import embed_texts
from retrieval import retrieve_chunks
from fastapi import UploadFile, File
from rewrite import rewrite_query
import json


load_dotenv()

app = FastAPI(title="Chatbot API", version="0.1.0")

# Very simple in-memory chat history:
# { session_id: [ {"role": "user"/"assistant", "content": "..."} , ... ] }
CHAT_HISTORY: dict[str, list[dict[str, str]]] = {}

MAX_TURNS = 12  # keep last 12 messages total (6 user+assistant pairs)

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


class IngestRequest(BaseModel):
    name: str
    text: str
    metadata: dict = {}


@app.on_event("startup")
async def _startup():
    await init_pool()


@app.on_event("shutdown")
async def _shutdown():
    await close_pool()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/db-health")
async def db_health():
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT now() AS ts;")
    return {"db": "ok", "time": row["ts"].isoformat()}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    t0 = time.time()
    msg = req.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")

    # Embed user query
    history = CHAT_HISTORY.get(req.sessionId, [])
    # 1) Rewrite latest message into a standalone retrieval query
    retrieval_query = await rewrite_query(msg, history)
    print("RETRIEVAL_QUERY:", retrieval_query)

    use_context = False
    hits = []
    best_distance = 0
    if retrieval_query != "NO_QUERY":
        # 2) Embed rewritten query
        q_emb = (await embed_texts([retrieval_query]))[0]
        # 3) Retrieve relevant chunks
        hits = await retrieve_chunks(q_emb, k=4)
        best_distance = float(hits[0]["distance"]) if hits else None
        USE_CONTEXT_THRESHOLD = 0.62  # tweak later
        use_context = hits and best_distance < USE_CONTEXT_THRESHOLD

    # Build context + sources
    context_parts = []
    sources = []

    if use_context:
        for r in hits:
            context_parts.append(f"[chunk {r['id']}] {r['content']}")
            sources.append({
                "chunkId": r["id"],
                "documentId": r["document_id"],
                "chunkIndex": r["chunk_index"],
                "metadata": r["metadata"],
                "distance": float(r["distance"]),
                "preview": r["content"][:200]
            })

    context_text = "\n\n".join(context_parts)

    # answer
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    app_url = os.getenv("OPENROUTER_APP_URL", "http://localhost:3000")
    app_name = os.getenv("OPENROUTER_APP_NAME", "Local Dev")

    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set")

    # Build prompt/message
    system_prompt = (
        "You are the embeded ai assistant for a computer science student named Amanda Li in her personal portfolio website.\n"
        "Your purpose is to help visitors understand her accomplishments, skills, values, projects, and about her'.\n"
        "Tone: Clear, friendly, and avoid sounding like generic ChatGPT. No emojis or markdown. Write in clean, plain text."
        "If CONTEXT is provided, use it as context and cite chunks like [chunk 123].\n"
        "If no CONTEXT is provided, ask for further clarification and DO NOT make anything up.\n"
        "You have access to the conversation history provided in the messages list.\n"
        "Use it to stay consistent and answer questions about what the user said earlier.\n"
        "Never invent citations. Keep answers within 200 tokens."
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)

    # Inject retrieved knowledge
    if use_context:
        messages.append({
            "role": "system",
            "content": f"CONTEXT:\n{context_text}"
        })

    messages.append({"role": "user", "content": msg})

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": app_url,   # optional attribution
                "X-Title": app_name,       # optional attribution
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 200,
            },
        )

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"OpenRouter error: {r.text}")

    data = r.json()
    answer = data["choices"][0]["message"]["content"]
    # answer = f"{answer}\n\n(debug: used_context={use_context}, best_distance={best_distance})"

    # Update memory
    new_history = history + [
        {"role": "user", "content": msg},
        {"role": "assistant", "content": answer},
    ]

    # Trim old history (keep only most recent MAX_TURNS messages)
    CHAT_HISTORY[req.sessionId] = new_history[-MAX_TURNS:]

    return ChatResponse(
        messageId=str(uuid.uuid4()),
        answer=answer,
        sources=sources,
        latencyMs=int((time.time() - t0) * 1000),
    )


@app.post("/ingest")
async def ingest(req: IngestRequest):
    chunks = chunk_text(req.text)
    if not chunks:
        raise HTTPException(status_code=400, detail="text is empty")

    # 1) embed all chunks
    vectors = await embed_texts(chunks)

    pool = get_pool()
    async with pool.acquire() as conn:
        # 2) create a document row
        doc_row = await conn.fetchrow(
            "INSERT INTO documents (name) VALUES ($1) RETURNING id;",
            req.name,
        )
        document_id = doc_row["id"]

        # 3) insert chunks
        for i, (content, emb) in enumerate(zip(chunks, vectors)):
            await conn.execute(
                """
                INSERT INTO doc_chunks (document_id, chunk_index, content, metadata, embedding)
                VALUES ($1, $2, $3, $4, $5)
                """,
                document_id,
                i,
                content,
                json.dumps(req.metadata),
                emb,
            )

    return {"status": "ok", "documentId": document_id, "chunksInserted": len(chunks)}


@app.post("/ingest-file")
async def ingest_file(file: UploadFile = File(...)):
    name = file.filename or "uploaded"
    raw = await file.read()

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Only UTF-8 text files are supported for now")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="file is empty")

    vectors = await embed_texts(chunks)

    pool = get_pool()
    async with pool.acquire() as conn:
        doc_row = await conn.fetchrow(
            "INSERT INTO documents (name) VALUES ($1) RETURNING id;",
            name,
        )
        document_id = doc_row["id"]

        meta_json = json.dumps({"source": name})

        for i, (content, emb) in enumerate(zip(chunks, vectors)):
            await conn.execute(
                """
                INSERT INTO doc_chunks (document_id, chunk_index, content, metadata, embedding)
                VALUES ($1, $2, $3, $4::jsonb, $5)
                """,
                document_id,
                i,
                content,
                meta_json,
                emb,
            )

    return {"status": "ok", "documentId": document_id, "chunksInserted": len(chunks)}


@app.get("/chunks/{chunk_id}")
async def get_chunk(chunk_id: int):
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, document_id, chunk_index, content, metadata
            FROM doc_chunks
            WHERE id = $1
            """,
            chunk_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="chunk not found")

    return {
        "id": row["id"],
        "documentId": row["document_id"],
        "chunkIndex": row["chunk_index"],
        "content": row["content"],
        "metadata": row["metadata"],
    }
