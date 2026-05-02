#!/usr/bin/env python3
"""Script to ingest and process all documents."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_ingestion import DataIngestionPipeline
from src.embeddings import EmbeddingManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the complete data ingestion pipeline."""
    print("🚀 Starting data ingestion pipeline...")
    
    # Initialize pipeline
    pipeline = DataIngestionPipeline()
    embedding_manager = EmbeddingManager()
    
    # Step 1: Process documents
    print("📄 Processing documents...")
    chunks = pipeline.ingest_documents()
    
    if not chunks:
        print("⚠️  No documents found to process. Add files to data/raw/ directory.")
        return
    
    # Step 2: Save processed chunks
    print("💾 Saving processed chunks...")
    pipeline.save_processed_data(chunks)
    
    # Step 3: Create embeddings and add to vector database
    print("🔍 Creating embeddings and adding to vector database...")
    embedding_manager.add_chunks_to_db(chunks)
    
    # Step 4: Show statistics
    stats = embedding_manager.get_collection_stats()
    print(f"✅ Data ingestion complete!")
    print(f"   - Total chunks: {stats['total_chunks']}")
    print(f"   - Embedding model: {stats['embedding_model']}")
    print(f"   - Collection: {stats['collection_name']}")

if __name__ == "__main__":
    main()
