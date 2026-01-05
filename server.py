"""
FastAPI Server for LangGraph Multi-Database Agent
Provides HTTP endpoints with SSE streaming
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging

# Import the agent code (save previous artifact as agent.py)
from agent import process_query_stream

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="LangGraph Multi-Database Agent API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST MODELS
# =============================================================================

class QueryRequest(BaseModel):
    query: str
    collection_name: Optional[str] = None


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "LangGraph Multi-Database Agent",
        "version": "1.0.0"
    }


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    Query the agent with streaming response
    
    Args:
        query: User's question
        collection_name: Optional vector store collection
    
    Returns:
        Server-Sent Events stream
    """
    
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    logger.info(f"Processing query: {request.query[:100]}...")
    
    return StreamingResponse(
        process_query_stream(
            user_query=request.query,
            collection_name=request.collection_name
        ),
        media_type="text/event-stream"
    )


@app.post("/query/simple")
async def simple_query(request: QueryRequest):
    """
    Non-streaming query endpoint (collects full response)
    
    Returns:
        Complete response as JSON
    """
    
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    collected_response = ""
    tool_calls = []
    
    try:
        async for chunk in process_query_stream(
            user_query=request.query,
            collection_name=request.collection_name
        ):
            if chunk.startswith("data: "):
                import json
                data = json.loads(chunk[6:])
                
                if data['type'] == 'content':
                    collected_response += data['data']
                elif data['type'] == 'tool_call':
                    tool_calls.append(data)
        
        return {
            "success": True,
            "response": collected_response,
            "tool_calls": tool_calls
        }
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# STARTUP/SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting LangGraph Agent API...")
    logger.info("Databases: PostgreSQL, MongoDB, Qdrant")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down LangGraph Agent API...")


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )