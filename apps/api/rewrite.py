import os
import httpx

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

async def rewrite_query(
        latest_user_msg: str,
        history: list[dict[str, str]],
        max_history_messages: int = 10,
) -> str:
    """
    Uses the conversation context to rewrite the latest user message into
    a standalone semantic search query for retrieval.
    Returns a single-line query string.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    model = os.getenv("OPENROUTER_REWRITE_MODEL", "openai/gpt-4o-mini")

    # Keep only recent history to control cost and noise
    trimmed = history[-max_history_messages:] if history else []

    system = (
        "TASK: Rewrite the user's latest message into a standalone semantic SEARCH QUERY for retrieving relevant document chunks.\n"
        "You are NOT the assistant and must NOT reply to the user. The user is talking to a chatbot in a personal portfolio that provides information about Amanda Li\n"
        "Rules:\n"
        "- Output ONLY the search query.\n"
        "- Do NOT include polite phrases (e.g., 'you're welcome', 'sure') unless relevant.\n"
        "- Do NOT include full sentences addressed to the user.\n"
        "- Use keywords and short phrases. Consider and keep all relevant context from the user's message and conversation history\n"
        "- If the message is vague (e.g., 'tell me more', 'explain that'), use the conversation history to infer the topic and produce a specific query.\n"
        "- If the message is ONLY gratitude or small talk with no information needed and no topic in history, output: NO_QUERY\n"
        "Examples:\n"
        "User: 'tell me more' (after talking about Amanda's favourite colour)\n"
        "Output: Amanda colour pink color favourite likes\n"
        "User: 'thanks'\n"
        "Output: NO_QUERY"
    )

    msgs = [{"role": "system", "content": system}]
    msgs.extend(trimmed)
    msgs.append({"role": "user", "content": latest_user_msg})

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            OPENROUTER_CHAT_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": msgs,
                "temperature": 0.0,
                "max_tokens": 60,
            },
        )

    if r.status_code >= 400:
        raise RuntimeError(f"OpenRouter rewrite error {r.status_code}: {r.text}")

    data = r.json()
    out = data["choices"][0]["message"]["content"].strip()

    # Safety cleanup: take first line only, remove surrounding quotes
    out = out.splitlines()[0].strip().strip('"').strip("'")

    # Fallback: if the model returned something empty, use the original
    return out if out else latest_user_msg
