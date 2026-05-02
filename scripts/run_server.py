#!/usr/bin/env python3
"""Script to run the MCP server."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import uvicorn
from config.settings import settings

def main():
    """Run the MCP server."""
    print("🌐 Starting MCP server...")
    print(f"   - Host: {settings.MCP_SERVER_HOST}")
    print(f"   - Port: {settings.MCP_SERVER_PORT}")
    print(f"   - API Key: {settings.API_KEY}")
    
    uvicorn.run(
        "src.mcp_server:app",
        host=settings.MCP_SERVER_HOST,
        port=settings.MCP_SERVER_PORT,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
