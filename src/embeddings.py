"""Embedding generation and vector database management."""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import numpy as np
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embeddings and vector database operations."""
    
    def __init__(self):
        # Initialize embedding model
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(settings.CHROMA_DB_PATH),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_chunks",
            metadata={"description": "Personal document chunks for RAG"}
        )
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def add_chunks_to_db(self, chunks: List[Dict[str, Any]]) -> None:
        """Add chunks with embeddings to the vector database."""
        if not chunks:
            logger.warning("No chunks to add to database")
            return
        
        # Extract texts and metadata
        texts = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Create unique IDs
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Add to ChromaDB
        logger.info(f"Adding {len(chunks)} chunks to vector database")
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Successfully added {len(chunks)} chunks to database")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks based on query."""
        k = k or settings.TOP_K_RESULTS
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_metadata
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": 1 - results["distances"][0][i],  # Convert distance to similarity
                "id": results["ids"][0][i]
            })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "embedding_model": settings.EMBEDDING_MODEL,
            "collection_name": "document_chunks"
        }
    
    def clear_database(self) -> None:
        """Clear all data from the vector database."""
        logger.warning("Clearing vector database")
        self.chroma_client.reset()
        self.collection = self.chroma_client.get_or_create_collection(
            name="document_chunks",
            metadata={"description": "Personal document chunks for RAG"}
        )

# Example usage
if __name__ == "__main__":
    import json
    
    # Load processed chunks
    chunks_file = settings.PROCESSED_DATA_DIR / "processed_chunks.json"
    if chunks_file.exists():
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Initialize embedding manager and add chunks
        embedding_manager = EmbeddingManager()
        embedding_manager.add_chunks_to_db(chunks)
        
        # Test similarity search
        results = embedding_manager.similarity_search("What is the main topic?")
        print(f"Found {len(results)} similar chunks")
        for result in results:
            print(f"Similarity: {result['similarity']:.3f}")
            print(f"Content: {result['content'][:100]}...")
            print("---")
    else:
        print("No processed chunks found. Run data ingestion first.")
