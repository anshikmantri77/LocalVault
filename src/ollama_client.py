"""Ollama API client for local inference using httpx (async)."""

import httpx
import json
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

async def generate(prompt: str, model: str = None, temperature: float = 0.7):
    """Call local Ollama API for generation using async httpx."""
    model = model or settings.OLLAMA_MODEL
    url = f"{settings.OLLAMA_BASE_URL}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            content = result.get("response", "")
            if not content:
                logger.warning(f"Ollama returned empty response for model {model}")
                
            return {
                "content": content,
                "success": True
            }
    except Exception as e:
        logger.error(f"Ollama generation failed: {e}")
        return {
            "content": f"Ollama Error: {str(e)}",
            "success": False
        }
