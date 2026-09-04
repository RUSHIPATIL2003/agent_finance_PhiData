"""FastAPI backend service exposing REST endpoints for the Financial AI Agent."""

import asyncio
from contextlib import asynccontextmanager
import logging
import uuid
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.rag_agent.config import get_settings
from src.rag_agent.schemas import (
    ChatMessage,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    SessionClearResponse,
)
from src.rag_agent.agent import run_financial_agent, create_model_instance
from src.rag_agent.memory import chat_memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("rag_agent.api")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle for connection pools."""
    logger.info("Starting up FastAPI application and initializing Redis episodic memory pool...")
    await chat_memory.initialize()
    yield
    logger.info("Shutting down FastAPI application and closing Redis connection pool...")
    await chat_memory.close()


app = FastAPI(
    title="Financial AI Agent API",
    description="High-performance backend serving the multi-agent financial intelligence assistant with Redis episodic memory.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for frontend and external callers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    response_model=dict,
    tags=["General"],
    summary="Root Service Info",
)
async def root():
    """Root endpoint providing service metadata."""
    return {
        "service": "Financial AI Multi-Agent Backend",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health & Readiness Check",
)
async def health_check():
    """Verify backend health, active LLM configuration, and Redis episodic memory status."""
    try:
        _, model_info = create_model_instance()
        provider_name = model_info.split("(")[0].strip() if "(" in model_info else "Configured"
        redis_alive = await chat_memory.ping()
        redis_status_text = "connected" if redis_alive else "disconnected (operating in fallback mode)"

        return HealthResponse(
            status="healthy",
            version="0.1.0",
            agent_ready=True,
            model_provider=provider_name,
            model_id=model_info,
            redis_connected=redis_alive,
            redis_status=redis_status_text,
        )
    except Exception as e:
        logger.error("Health check error: %s", e)
        return HealthResponse(
            status="degraded",
            version="0.1.0",
            agent_ready=False,
            model_provider="unknown",
            model_id=str(e),
            redis_connected=False,
            redis_status=f"error: {str(e)}",
        )


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
    tags=["Chat"],
    summary="Financial Chat Completion with Episodic Memory",
)
async def chat_endpoint(payload: ChatRequest):
    """Process user queries through the multi-agent financial coordinator with Redis episodic caching."""
    query = payload.message.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message prompt cannot be empty.",
        )

    session_id = payload.session_id or f"sess_{uuid.uuid4().hex[:8]}"
    logger.info("Processing query for session [%s]: %s", session_id, query[:80])

    # 1. Resiliently fetch conversation history from Redis
    try:
        history = await chat_memory.get_history(session_id)
        if history:
            logger.info("Retrieved %d historical messages for session [%s]", len(history), session_id)
    except Exception as mem_err:
        logger.warning(
            "Failed to retrieve chat history from Redis for session [%s]: %s. Falling back to stateless turn.",
            session_id,
            mem_err,
        )
        history = []

    # 2. Execute financial agent with memory context
    try:
        response_text, model_used = await asyncio.to_thread(
            run_financial_agent,
            query=query,
            history=history,
        )
        logger.info("Successfully generated response for session [%s]", session_id)
    except Exception as e:
        logger.exception("Error executing financial agent for query: %s", query)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(e)}",
        )

    # 3. Asynchronously push the new user message and model response back into Redis
    try:
        saved = await chat_memory.add_turn(
            session_id=session_id,
            user_message=query,
            assistant_response=response_text,
        )
        if saved:
            logger.debug("Successfully updated Redis episodic memory for session [%s]", session_id)
    except Exception as save_err:
        logger.warning(
            "Failed to persist conversation turn into Redis for session [%s]: %s",
            session_id,
            save_err,
        )

    return ChatResponse(
        response=response_text,
        session_id=session_id,
        status="success",
        model_used=model_used,
    )


@app.get(
    "/api/chat/history/{session_id}",
    response_model=ChatHistoryResponse,
    tags=["Chat"],
    summary="Retrieve Session Chat History",
)
async def get_session_history(session_id: str):
    """Retrieve all cached messages in episodic memory for a specific session."""
    try:
        history = await chat_memory.get_history(session_id)
        messages = [
            ChatMessage(
                role=item.get("role", "user"),
                content=str(item.get("content", "")),
                timestamp=item.get("timestamp"),
            )
            for item in history
        ]
        return ChatHistoryResponse(
            session_id=session_id,
            messages=messages,
            total_messages=len(messages),
            status="success",
        )
    except Exception as exc:
        logger.warning("Error retrieving history for session [%s]: %s", session_id, exc)
        return ChatHistoryResponse(
            session_id=session_id,
            messages=[],
            total_messages=0,
            status="error",
        )


@app.delete(
    "/api/chat/history/{session_id}",
    response_model=SessionClearResponse,
    tags=["Chat"],
    summary="Clear Session Chat History",
)
async def clear_session_history(session_id: str):
    """Explicitly purge cached conversation memory for a specific session."""
    try:
        cleared = await chat_memory.clear_session(session_id)
        return SessionClearResponse(
            session_id=session_id,
            cleared=cleared,
            message="Session memory successfully cleared" if cleared else "Session memory could not be cleared or was empty",
            status="success",
        )
    except Exception as exc:
        logger.warning("Error clearing history for session [%s]: %s", session_id, exc)
        return SessionClearResponse(
            session_id=session_id,
            cleared=False,
            message=f"Failed to clear session: {str(exc)}",
            status="error",
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Standardized JSON response for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status": "error"},
    )
