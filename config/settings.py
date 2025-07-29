"""Configuration settings for the LLM chatbot project."""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = DATA_DIR / "models"
    
    # Ollama configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    BASE_MODEL: str = "llama3:8b"
    FINE_TUNED_MODEL: str = "personal-assistant"
    
    # Vector database
    CHROMA_DB_PATH: Path = DATA_DIR / "chroma_db"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Server configuration
    MCP_SERVER_HOST: str = "localhost"
    MCP_SERVER_PORT: int = 8000
    API_KEY: str = "your-secret-api-key-here"
    
    # Processing parameters
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 50
    TOP_K_RESULTS: int = 5
    
    # Fine-tuning parameters
    LORA_RANK: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.1
    LEARNING_RATE: float = 2e-4
    BATCH_SIZE: int = 4
    NUM_EPOCHS: int = 3
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
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
