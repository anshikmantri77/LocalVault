
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

from src.embeddings import EmbeddingManager
from config.settings import settings

def test():
    manager = EmbeddingManager()
    query = "PharmEasy"
    print(f"Testing search for: '{query}'")
    
    results = manager.similarity_search(query, k=5)
    print(f"Found {len(results)} results")
    
    for i, res in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Score: {res.get('similarity', 'N/A')}")
        print(f"  File: {res.get('metadata', {}).get('filename', 'Unknown')}")
        print(f"  Text: {res.get('content')[:100]}...")

if __name__ == "__main__":
    test()
