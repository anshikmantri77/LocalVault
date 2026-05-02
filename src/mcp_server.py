"""
FastAPI-based backend server for LocalVault.

Fixes applied:
  7. Service Instability: /health returns 503 while loading (not 200),
     preventing race-condition queries; upload endpoint deduplicates files
     so re-uploading the same PDF doesn't double-index everything.
  8. Legacy Code Clutter: removed dead hf_client import, removed unused
     AVAILABLE_MODELS references, removed duplicate /clear endpoint logic.
"""

import logging
import hashlib
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag_pipeline import EnhancedRAGPipeline
from src.data_ingestion import DataIngestionPipeline
from src.embeddings import EmbeddingManager
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    model: Optional[str] = None
    include_sources: bool = True
    k: int = 5
    use_conversation_context: bool = True
    temperature: float = 0.1


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


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="LocalVault API",
    description="Private RAG-powered document assistant",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components — set to None until startup finishes
rag_pipeline: Optional[EnhancedRAGPipeline] = None
data_pipeline: Optional[DataIngestionPipeline] = None
embedding_manager: Optional[EmbeddingManager] = None
is_ready: bool = False


# ---------------------------------------------------------------------------
# Startup — load heavy models in background
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    async def load_components():
        global rag_pipeline, data_pipeline, embedding_manager, is_ready
        logger.info("🚀 Loading AI components in background…")
        try:
            _em = EmbeddingManager()
            rag_pipeline = EnhancedRAGPipeline(embedding_manager=_em)
            data_pipeline = DataIngestionPipeline(embedding_manager=_em)
            embedding_manager = _em
            is_ready = True
            logger.info("✅ AI components loaded and ready.")
        except Exception as exc:
            logger.error("❌ Failed to load AI components: %s", exc)

    asyncio.create_task(load_components())


def _require_ready():
    """Dependency that raises 503 while system is still loading."""
    if not is_ready:
        raise HTTPException(
            status_code=503,
            detail="System is still initialising. Please wait ~30 seconds and retry.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Returns 503 while loading so the frontend shows the real state."""
    if not is_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "initializing", "version": "2.0.0", "model": "Loading…", "database_stats": {}},
        )
    db_stats = embedding_manager.get_collection_stats()
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        model=rag_pipeline.model,
        database_stats=db_stats,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    token: str = Depends(verify_token),
    _: None = Depends(_require_ready),
):
    """Main chat endpoint."""
    try:
        if request.model:
            rag_pipeline.set_model(request.model)

        result = await rag_pipeline.chat_with_memory(
            query=request.query,
            include_sources=request.include_sources,
            k=request.k,
            use_conversation_context=request.use_conversation_context,
            temperature=request.temperature,
        )
        return ChatResponse(**result)
    except Exception as exc:
        logger.error("Chat error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    clear_first: bool = Form(False),
    token: str = Depends(verify_token),
    _: None = Depends(_require_ready),
):
    """
    Upload and process documents.
    - If clear_first=True the entire vector DB is wiped before indexing.
    - Duplicate files (same name + same content hash) are skipped so
      re-uploading doesn't double-index — fixes Issue #7.
    """
    try:
        if clear_first:
            logger.info("Clearing vector database before upload.")
            embedding_manager.clear_database()
            rag_pipeline.memory.clear()

        files_processed = 0

        for file in files:
            raw_bytes = await file.read()
            file_hash = hashlib.md5(raw_bytes).hexdigest()
            final_path = settings.RAW_DATA_DIR / file.filename

            # Skip if identical file already exists
            if final_path.exists():
                existing_hash = hashlib.md5(final_path.read_bytes()).hexdigest()
                if existing_hash == file_hash:
                    logger.info("Skipping duplicate file: %s", file.filename)
                    continue

            final_path.write_bytes(raw_bytes)
            files_processed += 1
            logger.info("Saved uploaded file: %s", file.filename)

        # Re-process all docs in raw dir and rebuild the index
        chunks = data_pipeline.ingest_documents()
        data_pipeline.save_processed_data(chunks)

        # Clear existing vectors and re-add (avoids stale duplicates in DB)
        embedding_manager.clear_database()
        embedding_manager.add_chunks_to_db(chunks)
        total_chunks = len(chunks)

        return UploadResponse(
            message=f"Successfully processed {files_processed} new file(s)",
            files_processed=files_processed,
            chunks_created=total_chunks,
            success=True,
        )

    except Exception as exc:
        logger.error("Upload error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/embeddings/stats")
async def get_embedding_stats(
    token: str = Depends(verify_token),
    _: None = Depends(_require_ready),
):
    try:
        return embedding_manager.get_collection_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/embeddings/clear")
async def clear_embeddings(
    token: str = Depends(verify_token),
    _: None = Depends(_require_ready),
):
    try:
        embedding_manager.clear_database()
        return {"message": "Vector database cleared."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/conversation/history")
async def get_conversation_history(
    token: str = Depends(verify_token),
    _: None = Depends(_require_ready),
):
    return {
        "history": rag_pipeline.memory.history,
        "count": len(rag_pipeline.memory.history),
    }


@app.post("/conversation/clear")
async def clear_conversation_history(
    token: str = Depends(verify_token),
    _: None = Depends(_require_ready),
):
    rag_pipeline.memory.clear()
    return {"message": "Conversation history cleared."}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "src.mcp_server:app",
        host=settings.MCP_SERVER_HOST,
        port=settings.MCP_SERVER_PORT,
        reload=False,   # reload=True causes duplicate startup tasks
        log_level="info",
    )
