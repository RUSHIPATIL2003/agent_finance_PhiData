"""FastAPI backend service exposing REST endpoints for the Financial AI Agent."""

import logging
import uuid
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.rag_agent.config import get_settings
from src.rag_agent.schemas import ChatRequest, ChatResponse, HealthResponse, ErrorResponse
from src.rag_agent.agent import run_financial_agent, create_model_instance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("rag_agent.api")

settings = get_settings()

app = FastAPI(
    title="Financial AI Agent API",
    description="High-performance backend serving the multi-agent financial intelligence assistant.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
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
    """Verify backend health and active LLM configuration."""
    try:
        _, model_info = create_model_instance()
        provider_name = model_info.split("(")[0].strip() if "(" in model_info else "Configured"
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            agent_ready=True,
            model_provider=provider_name,
            model_id=model_info,
        )
    except Exception as e:
        logger.error("Health check error: %s", e)
        return HealthResponse(
            status="degraded",
            version="0.1.0",
            agent_ready=False,
            model_provider="unknown",
            model_id=str(e),
        )


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
    tags=["Chat"],
    summary="Financial Chat Completion",
)
async def chat_endpoint(payload: ChatRequest):
    """Process user queries through the multi-agent financial coordinator."""
    query = payload.message.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message prompt cannot be empty.",
        )

    session_id = payload.session_id or f"sess_{uuid.uuid4().hex[:8]}"
    logger.info("Processing query for session [%s]: %s", session_id, query[:80])

    try:
        response_text, model_used = run_financial_agent(query)
        logger.info("Successfully generated response for session [%s]", session_id)
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            status="success",
            model_used=model_used,
        )
    except Exception as e:
        logger.exception("Error executing financial agent for query: %s", query)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(e)}",
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Standardized JSON response for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status": "error"},
    )
