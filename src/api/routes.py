"""FastAPI routes for the RAG Agent."""

import json
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.api import models
from src.agent.orchestrator import get_agent
from src.retrieval.vector_store import get_chunk_count

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None
    pet_name: str = ""
    pet_breed: str = ""
    pet_age: str = ""


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    citations: list[dict] = []
    triage: dict = {}
    requires_confirmation: bool = False
    source: str = "local"


class ConversationMeta(BaseModel):
    id: str
    title: str
    pet_name: str
    updated_at: float


@router.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": get_chunk_count(),
        "conversations": len(models.list_conversations()),
    }


@router.post("/chat")
def chat(req: ChatRequest) -> ChatResponse:
    conv_id = req.conversation_id or str(uuid.uuid4())[:8]

    # Ensure conversation exists
    if not models.get_conversation(conv_id):
        models.create_conversation(
            conv_id,
            pet_name=req.pet_name,
            pet_breed=req.pet_breed,
            pet_age=req.pet_age,
        )

    # Save user message
    models.save_message(conv_id, "user", req.message)

    # Build context with pet profile
    context = ""
    if req.pet_name:
        context += f"宠物名: {req.pet_name}。"
    if req.pet_breed:
        context += f"品种: {req.pet_breed}。"
    if req.pet_age:
        context += f"年龄: {req.pet_age}。"

    full_query = f"[宠物档案] {context}\n\n{req.message}" if context else req.message

    # 🏗️ Get structured agent response
    agent = get_agent()
    result = agent.chat_structured(full_query)
    structured = result.to_dict()

    # Save assistant response with structured metadata
    import json as _json
    models.save_message(
        conv_id, "assistant",
        structured["answer"],
        metadata=structured,
    )

    return ChatResponse(
        conversation_id=conv_id,
        response=structured["answer"],
        citations=structured.get("citations", []),
        triage=structured.get("triage", {}),
        requires_confirmation=structured.get("requires_confirmation", False),
        source=structured.get("source", "local"),
    )


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """🏗️ SSE streaming chat endpoint — yields tokens as they arrive."""
    conv_id = req.conversation_id or str(uuid.uuid4())[:8]

    if not models.get_conversation(conv_id):
        models.create_conversation(
            conv_id,
            pet_name=req.pet_name,
            pet_breed=req.pet_breed,
            pet_age=req.pet_age,
        )
    models.save_message(conv_id, "user", req.message)

    context = ""
    if req.pet_name:
        context += f"宠物名: {req.pet_name}。"
    if req.pet_breed:
        context += f"品种: {req.pet_breed}。"
    if req.pet_age:
        context += f"年龄: {req.pet_age}。"

    full_query = f"[宠物档案] {context}\n\n{req.message}" if context else req.message

    async def generate():
        agent = get_agent()
        for event in agent.chat_stream(full_query):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/conversations")
def list_convs():
    convs = models.list_conversations()
    return [
        ConversationMeta(
            id=c["id"],
            title=c["title"] or f"对话 {c['id']}",
            pet_name=c["pet_name"],
            updated_at=c["updated_at"],
        )
        for c in convs
    ]


@router.get("/conversations/{conv_id}/messages")
def get_conv_messages(conv_id: str):
    conv = models.get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    msgs = models.get_messages(conv_id)
    return {
        "conversation": dict(conv),
        "messages": [dict(m) for m in msgs],
    }


@router.delete("/conversations/{conv_id}")
def delete_conv(conv_id: str):
    models.delete_conversation(conv_id)
    return {"ok": True}
