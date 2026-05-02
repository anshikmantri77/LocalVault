"""Comprehensive tests for the chatbot pipeline."""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_ingestion import DocumentProcessor, DataIngestionPipeline
from src.embeddings import EmbeddingManager
from src.rag_pipeline import RAGPipeline
from config.settings import settings

class TestDocumentProcessor(unittest.TestCase):
    """Test document processing functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.processor = DocumentProcessor()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_text_processing(self):
        """Test text file processing."""
        # Create test text file
        test_file = self.temp_dir / "test.txt"
        test_content = "This is a test document with some content."
        
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        # Process file
        result = self.processor.process_file(test_file)
        self.assertEqual(result.strip(), test_content)
    
    def test_chunking(self):
        """Test text chunking functionality."""
        text = "This is a test. " * 100  # Long text
        metadata = {"source": "test", "filename": "test.txt"}
        
        chunks = self.processor.chunk_text(text, metadata)
        
        # Verify chunks were created
        self.assertGreater(len(chunks), 0)
        
        # Verify chunk structure
        for chunk in chunks:
            self.assertIn("content", chunk)
            self.assertIn("metadata", chunk)
            self.assertIn("chunk_index", chunk["metadata"])

class TestEmbeddingManager(unittest.TestCase):
    """Test embedding and vector database functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        # Temporarily override database path
        self.original_path = settings.CHROMA_DB_PATH
        settings.CHROMA_DB_PATH = self.temp_dir / "test_chroma"
        
        self.embedding_manager = EmbeddingManager()
    
    def tearDown(self):
        """Clean up test environment."""
        settings.CHROMA_DB_PATH = self.original_path
        shutil.rmtree(self.temp_dir)
    
    def test_embedding_generation(self):
        """Test embedding generation."""
        text = "This is a test document."
        embedding = self.embedding_manager.generate_embedding(text)
        
        # Verify embedding properties
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)
        self.assertTrue(all(isinstance(x, float) for x in embedding))
    
    def test_similarity_search(self):
        """Test similarity search functionality."""
        # Add test chunks
        test_chunks = [
            {
                "content": "This is about artificial intelligence.",
                "metadata": {"filename": "ai.txt", "chunk_index": 0}
            },
            {
                "content": "This discusses machine learning algorithms.",
                "metadata": {"filename": "ml.txt", "chunk_index": 0}
            }
        ]
        
        self.embedding_manager.add_chunks_to_db(test_chunks)
        
        # Test search
        results = self.embedding_manager.similarity_search("machine learning", k=1)
        
        # Verify results
        self.assertEqual(len(results), 1)
        self.assertIn("machine learning", results[0]["content"].lower())

class TestRAGPipeline(unittest.TestCase):
    """Test RAG pipeline functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # This would require a running Ollama instance
        # For testing, we might want to mock the Ollama client
        pass
    
    def test_prompt_building(self):
        """Test RAG prompt construction."""
        # This test can be implemented without external dependencies
        pass

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test data directory
        self.test_raw_dir = self.temp_dir / "raw"
        self.test_raw_dir.mkdir()
        
        # Create test files
        self.create_test_files()
    
    def tearDown(self):
        """Clean up integration test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_files(self):
        """Create test files for integration testing."""
        # Create test text file
        (self.test_raw_dir / "test.txt").write_text(
            "This is a test document about artificial intelligence and machine learning."
        )
        
        # Create test CSV
        import pandas as pd
        df = pd.DataFrame({
            "name": ["Project A", "Project B"],
            "status": ["Active", "Completed"],
            "description": ["AI research project", "ML model deployment"]
        })
        df.to_csv(self.test_raw_dir / "projects.csv", index=False)
    
    def test_full_pipeline(self):
        """Test the complete ingestion to retrieval pipeline."""
        # Temporarily override settings
        original_raw_dir = settings.RAW_DATA_DIR
        settings.RAW_DATA_DIR = self.test_raw_dir
        
        try:
            # Test data ingestion
            pipeline = DataIngestionPipeline()
            chunks = pipeline.ingest_documents()
            
            # Verify chunks were created
            self.assertGreater(len(chunks), 0)
            
            # Test that all files were processed
            processed_files = set(chunk["metadata"]["filename"] for chunk in chunks)
            self.assertIn("test.txt", processed_files)
            self.assertIn("projects.csv", processed_files)
            
        finally:
            settings.RAW_DATA_DIR = original_raw_dir

if __name__ == "__main__":
    # Run specific test categories
    import argparse
    
    parser = argparse.ArgumentParser(description="Run chatbot tests")
    parser.add_argument("--category", choices=["unit", "integration", "all"], 
                       default="all", help="Test category to run")
    args = parser.parse_args()
    
    if args.category == "unit":
        suite = unittest.TestLoader().loadTestsFromTestCase(TestDocumentProcessor)
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEmbeddingManager))
    elif args.category == "integration":
        suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegration)
    else:
        suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
