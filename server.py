"""
FastAPI Server for Multi-Database Agent (OpenAI version with Conversation History)
Provides REST API and SSE streaming endpoints with session management
"""

import asyncio
import logging
from typing import Optional, Dict
from contextlib import asynccontextmanager
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import the OpenAI agent and conversation module
from agent import (
    process_query_stream,
    process_query,
    postgres_service,
    mongo_service,
    vector_service,
    openai_client,
    execute_tool,
    TOOLS,
    Config,
    build_system_prompt,
    logger
)

from conversationModule import (
    ConversationHistory,
    ConversationManager,
    process_query_stream_with_history,
    process_query_with_history
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# =============================================================================
# GLOBAL CONVERSATION MANAGER
# =============================================================================

conversation_manager = ConversationManager()

# =============================================================================
# LIFESPAN MANAGEMENT
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("🚀 Starting FastAPI server with conversation history...")
    logger.info("✅ Database connections established")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down FastAPI server...")
    postgres_service.close()
    mongo_service.close()
    logger.info("✅ Cleanup complete")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Multi-Database Agent API with History (OpenAI)",
    description="Query PostgreSQL, MongoDB, and vector stores using natural language with conversation history",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., description="User's natural language query", min_length=1)
    collection_name: Optional[str] = Field(None, description="Optional vector store collection for context")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Show me product with code 123",
                "collection_name": None,
                "session_id": None
            }
        }


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    success: bool = Field(..., description="Whether the query was successful")
    response: str = Field(..., description="Agent's response text")
    tool_calls: list = Field(default_factory=list, description="List of tool calls made")
    timestamp: str = Field(..., description="ISO timestamp of response")
    session_id: str = Field(..., description="Session ID for this conversation")
    conversation_summary: Optional[dict] = Field(None, description="Summary of conversation")
    error: Optional[str] = Field(None, description="Error message if failed")


class SessionResponse(BaseModel):
    """Response model for session endpoints"""
    success: bool
    session_id: str
    message: Optional[str] = None
    conversation_summary: Optional[dict] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    postgres: str
    mongodb: str
    active_sessions: int
    timestamp: str


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_or_create_session(session_id: Optional[str] = None) -> tuple[str, ConversationHistory]:
    """
    Get existing conversation or create new one
    
    Returns:
        tuple: (session_id, conversation)
    """
    if session_id:
        conversation = conversation_manager.get_conversation(session_id)
        if conversation:
            logger.info(f"Retrieved existing session: {session_id}")
            return session_id, conversation
        else:
            logger.warning(f"Session {session_id} not found, creating new session")
    
    # Create new session
    new_session_id = str(uuid.uuid4())
    conversation = conversation_manager.create_conversation(
        conversation_id=new_session_id,
        system_prompt=build_system_prompt(),
        max_messages=20
    )
    logger.info(f"Created new session: {new_session_id}")
    
    return new_session_id, conversation


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multi-Database Agent API with Conversation History (OpenAI)",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/api/query",
            "stream": "/api/stream",
            "sessions": {
                "create": "POST /api/sessions",
                "get": "GET /api/sessions/{session_id}",
                "delete": "DELETE /api/sessions/{session_id}",
                "list": "GET /api/sessions"
            }
        },
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    # Check PostgreSQL
    try:
        postgres_service.execute("SELECT 1")
        postgres_status = "healthy"
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")
        postgres_status = "unhealthy"
    
    # Check MongoDB
    try:
        mongo_service.db.command("ping")
        mongo_status = "healthy"
    except Exception as e:
        logger.error(f"MongoDB health check failed: {e}")
        mongo_status = "unhealthy"
    
    overall_status = "healthy" if postgres_status == "healthy" and mongo_status == "healthy" else "degraded"
    
    return {
        "status": overall_status,
        "postgres": postgres_status,
        "mongodb": mongo_status,
        "active_sessions": len(conversation_manager.conversations),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/query", response_model=QueryResponse, tags=["Query"])
async def query_endpoint(request: QueryRequest):
    """
    Process query and return complete response (non-streaming) with conversation history
    
    This endpoint processes the query through the agent and returns
    the complete response once all processing is done.
    """
    try:
        logger.info(f"Received query: {request.query}")
        
        # Get or create session
        session_id, conversation = get_or_create_session(request.session_id)
        
        result = await process_query_with_history(
            user_query=request.query,
            conversation=conversation,
            collection_name=request.collection_name,
            openai_client=openai_client,
            vector_service=vector_service,
            execute_tool=execute_tool,
            TOOLS=TOOLS,
            Config=Config
        )
        
        if not result['success']:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Query processing failed",
                    "error": result.get('error'),
                    "error_type": result.get('error_type')
                }
            )
        
        # Add session_id to response
        result['session_id'] = session_id
        
        return QueryResponse(**result)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Internal server error",
                "error": str(e),
                "error_type": type(e).__name__
            }
        )


@app.post("/api/stream", tags=["Query"])
async def stream_endpoint(request: QueryRequest):
    """
    Process query with streaming response (SSE) and conversation history
    
    This endpoint streams the agent's response in real-time using
    Server-Sent Events (SSE) format while maintaining conversation context.
    
    Event types:
    - session_id: Session identifier (sent first)
    - tool_call: When agent calls a tool
    - tool_result: When tool returns results
    - content: Character-by-character response
    - complete: Processing finished
    - error: Error occurred
    """
    try:
        logger.info(f"Received streaming query: {request.query}")
        
        # Get or create session
        session_id, conversation = get_or_create_session(request.session_id)
        
        async def event_generator():
            """Generate SSE events"""
            try:
                # Send session_id first
                import json
                session_chunk = f"data: {json.dumps({'type': 'session_id', 'session_id': session_id})}\n\n"
                yield session_chunk
                
                # Stream the query response with history
                async for chunk in process_query_stream_with_history(
                    user_query=request.query,
                    conversation=conversation,
                    collection_name=request.collection_name,
                    openai_client=openai_client,
                    vector_service=vector_service,
                    execute_tool=execute_tool,
                    TOOLS=TOOLS,
                    Config=Config
                ):
                    yield chunk
                    
            except Exception as e:
                logger.error(f"Stream generation error: {e}", exc_info=True)
                import json
                error_chunk = f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                yield error_chunk
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    
    except Exception as e:
        logger.error(f"Stream endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Failed to initialize stream",
                "error": str(e)
            }
        )


# =============================================================================
# SESSION MANAGEMENT ENDPOINTS
# =============================================================================

@app.post("/api/sessions", response_model=SessionResponse, tags=["Sessions"])
async def create_session():
    """Create a new conversation session"""
    try:
        session_id = str(uuid.uuid4())
        conversation = conversation_manager.create_conversation(
            conversation_id=session_id,
            system_prompt=build_system_prompt(),
            max_messages=20
        )
        
        return SessionResponse(
            success=True,
            session_id=session_id,
            message="Session created successfully",
            conversation_summary=conversation.get_summary()
        )
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}", response_model=SessionResponse, tags=["Sessions"])
async def get_session(session_id: str):
    """Get conversation session details"""
    try:
        conversation = conversation_manager.get_conversation(session_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionResponse(
            success=True,
            session_id=session_id,
            conversation_summary=conversation.get_summary()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/sessions/{session_id}", response_model=SessionResponse, tags=["Sessions"])
async def delete_session(session_id: str):
    """Delete a conversation session"""
    try:
        conversation = conversation_manager.get_conversation(session_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Session not found")
        
        conversation_manager.delete_conversation(session_id)
        
        return SessionResponse(
            success=True,
            session_id=session_id,
            message="Session deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session deletion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions", tags=["Sessions"])
async def list_sessions():
    """List all active conversation sessions"""
    try:
        sessions = conversation_manager.list_conversations()
        return {
            "success": True,
            "count": len(sessions),
            "sessions": sessions
        }
    except Exception as e:
        logger.error(f"Session listing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions/{session_id}/clear", response_model=SessionResponse, tags=["Sessions"])
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    try:
        conversation = conversation_manager.get_conversation(session_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Session not found")
        
        conversation.clear()
        
        return SessionResponse(
            success=True,
            session_id=session_id,
            message="Session history cleared",
            conversation_summary=conversation.get_summary()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SCHEMA ENDPOINTS
# =============================================================================

@app.get("/api/schema/postgres", tags=["Schema"])
async def get_postgres_schema():
    """Get PostgreSQL database schema"""
    try:
        schema = postgres_service.get_schema()
        return {
            "success": True,
            "schema": schema,
            "database": "alpha-product-samir"
        }
    except Exception as e:
        logger.error(f"Schema fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/schema/mongodb", tags=["Schema"])
async def get_mongodb_schema():
    """Get MongoDB database schema"""
    try:
        schema = mongo_service.get_schema()
        return {
            "success": True,
            "schema": schema,
            "database": "alpha-kcc"
        }
    except Exception as e:
        logger.error(f"Schema fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return {
        "success": False,
        "error": "An unexpected error occurred",
        "detail": str(exc),
        "type": type(exc).__name__
    }


# =============================================================================
# STARTUP MESSAGE
# =============================================================================

@app.on_event("startup")
async def startup_message():
    """Print startup message"""
    logger.info("=" * 60)
    logger.info("🚀 Multi-Database Agent API with Conversation History")
    logger.info("=" * 60)
    logger.info("📚 Documentation: http://localhost:8000/docs")
    logger.info("🏥 Health Check: http://localhost:8000/health")
    logger.info("💬 Query Endpoint: POST http://localhost:8000/api/query")
    logger.info("🌊 Stream Endpoint: POST http://localhost:8000/api/stream")
    logger.info("📝 Sessions: http://localhost:8000/api/sessions")
    logger.info("=" * 60)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )