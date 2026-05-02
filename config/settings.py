"""Configuration settings for LocalVault."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = DATA_DIR / "models"

    # Ollama — 100% Local (Privacy First)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3.5:2b"   # High-performance local model

    # Vector database
    CHROMA_DB_PATH: Path = DATA_DIR / "chroma_db"
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"

    # Server
    MCP_SERVER_HOST: str = "localhost"
    MCP_SERVER_PORT: int = 8000
    API_KEY: str = "localvault-secret-key"

    # Processing — larger chunks preserve more sentence context for resumes
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 300
    TOP_K_RESULTS: int = 8

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

# Create necessary directories
for directory in [
    settings.DATA_DIR,
    settings.RAW_DATA_DIR,
    settings.PROCESSED_DATA_DIR,
    settings.MODELS_DIR,
    settings.CHROMA_DB_PATH,
]:
    directory.mkdir(parents=True, exist_ok=True)
