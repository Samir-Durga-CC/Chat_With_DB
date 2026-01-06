"""
FastAPI Server for Multi-Database Agent (OpenAI version)
Provides REST API and SSE streaming endpoints
"""

import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import the OpenAI agent
from agent import (
    process_query_stream,
    process_query,
    postgres_service,
    mongo_service,
    logger
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# =============================================================================
# LIFESPAN MANAGEMENT
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("🚀 Starting FastAPI server...")
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
    title="Multi-Database Agent API (OpenAI)",
    description="Query PostgreSQL, MongoDB, and vector stores using natural language",
    version="1.0.0",
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
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Show me product with code 123",
                "collection_name": None
            }
        }


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    success: bool = Field(..., description="Whether the query was successful")
    response: str = Field(..., description="Agent's response text")
    tool_calls: list = Field(default_factory=list, description="List of tool calls made")
    timestamp: str = Field(..., description="ISO timestamp of response")
    error: Optional[str] = Field(None, description="Error message if failed")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    postgres: str
    mongodb: str
    timestamp: str


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multi-Database Agent API (OpenAI)",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/api/query",
            "stream": "/api/stream"
        },
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    from datetime import datetime
    
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
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/query", response_model=QueryResponse, tags=["Query"])
async def query_endpoint(request: QueryRequest):
    """
    Process query and return complete response (non-streaming)
    
    This endpoint processes the query through the agent and returns
    the complete response once all processing is done.
    """
    try:
        logger.info(f"Received query: {request.query}")
        
        result = await process_query(
            user_query=request.query,
            collection_name=request.collection_name
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
    Process query with streaming response (SSE)
    
    This endpoint streams the agent's response in real-time using
    Server-Sent Events (SSE) format.
    
    Event types:
    - tool_call: When agent calls a tool
    - tool_result: When tool returns results
    - content: Character-by-character response
    - complete: Processing finished
    - error: Error occurred
    """
    try:
        logger.info(f"Received streaming query: {request.query}")
        
        async def event_generator():
            """Generate SSE events"""
            try:
                async for chunk in process_query_stream(
                    user_query=request.query,
                    collection_name=request.collection_name
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
    logger.info("🚀 Multi-Database Agent API (OpenAI)")
    logger.info("=" * 60)
    logger.info("📚 Documentation: http://localhost:8000/docs")
    logger.info("🏥 Health Check: http://localhost:8000/health")
    logger.info("💬 Query Endpoint: POST http://localhost:8000/api/query")
    logger.info("🌊 Stream Endpoint: POST http://localhost:8000/api/stream")
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