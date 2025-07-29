"""FastAPI-based MCP server for the chatbot backend."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import shutil

from src.rag_pipeline import EnhancedRAGPipeline
from src.data_ingestion import DataIngestionPipeline
from src.embeddings import EmbeddingManager
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class ChatRequest(BaseModel):
    query: str
    include_sources: bool = True
    k: int = 5
    use_conversation_context: bool = True

class ChatResponse(BaseModel):
    query: str
    response: str
    model: str
    success: bool
    sources: Optional[List[Dict[str, Any]]] = None
    context_used: Optional[int] = None
    error: Optional[str] = None

class UploadResponse(BaseModel):
    message: str
    files_processed: int
    chunks_created: int
    success: bool

class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    database_stats: Dict[str, Any]

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token."""
    if credentials.credentials != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Initialize FastAPI app
app = FastAPI(
    title="Personal LLM Chatbot API",
    description="MCP Server for personalized local chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
rag_pipeline = EnhancedRAGPipeline()
data_pipeline = DataIngestionPipeline()
embedding_manager = EmbeddingManager()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        db_stats = embedding_manager.get_collection_stats()
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            model=rag_pipeline.model_name,
            database_stats=db_stats
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unhealthy")

@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    token: str = Depends(verify_token)
):
    """Main chat endpoint."""
    try:
        result = rag_pipeline.chat_with_memory(
            query=request.query,
            include_sources=request.include_sources,
            k=request.k,
            use_conversation_context=request.use_conversation_context
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=UploadResponse)
async def upload_files(
    files: List[UploadFile] = File(...),
    token: str = Depends(verify_token)
):
    """Upload and process new documents."""
    try:
        files_processed = 0
        total_chunks = 0
        
        for file in files:
            # Save uploaded file temporarily
            temp_file = Path(tempfile.mkdtemp()) / file.filename
            
            with open(temp_file, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Move to raw data directory
            final_path = settings.RAW_DATA_DIR / file.filename
            shutil.move(str(temp_file), str(final_path))
            
            files_processed += 1
            logger.info(f"Saved uploaded file: {file.filename}")
        
        # Process new files
        chunks = data_pipeline.ingest_documents()
        data_pipeline.save_processed_data(chunks)
        
        # Add to vector database
        embedding_manager.add_chunks_to_db(chunks)
        total_chunks = len(chunks)
        
        return UploadResponse(
            message=f"Successfully processed {files_processed} files",
            files_processed=files_processed,
            chunks_created=total_chunks,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/embeddings/stats")
async def get_embedding_stats(token: str = Depends(verify_token)):
    """Get vector database statistics."""
    try:
        return embedding_manager.get_collection_stats()
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/embeddings/clear")
async def clear_embeddings(token: str = Depends(verify_token)):
    """Clear all embeddings from the database."""
    try:
        embedding_manager.clear_database()
        return {"message": "Vector database cleared successfully"}
    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/history")
async def get_conversation_history(token: str = Depends(verify_token)):
    """Get conversation history."""
    try:
        return {
            "history": rag_pipeline.memory.history,
            "count": len(rag_pipeline.memory.history)
        }
    except Exception as e:
        logger.error(f"History error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversation/clear")
async def clear_conversation_history(token: str = Depends(verify_token)):
    """Clear conversation history."""
    try:
        rag_pipeline.memory.history.clear()
        return {"message": "Conversation history cleared"}
    except Exception as e:
        logger.error(f"Clear history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "src.mcp_server:app",
        host=settings.MCP_SERVER_HOST,
        port=settings.MCP_SERVER_PORT,
        reload=True,
        log_level="info"
    )
