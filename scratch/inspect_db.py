import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.embeddings import EmbeddingManager
from config.settings import settings

def inspect_db():
    print("--- LocalVault Diagnostic: Database Inspection ---")
    em = EmbeddingManager()
    
    query = "Anshik Mantri Internship"
    results = em.similarity_search(query, k=10)
    
    print(f"\nQuery: '{query}'")
    print(f"Found {len(results)} results:")
    
    for i, res in enumerate(results):
        source = res['metadata'].get('filename', 'unknown')
        content = res['content'][:150].replace('\n', ' ')
        print(f"{i+1}. [{source}] (Score: {res['similarity']:.4f}) - {content}...")

if __name__ == "__main__":
    inspect_db()
