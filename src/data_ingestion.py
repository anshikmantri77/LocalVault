"""Data ingestion pipeline for processing various document formats."""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from PIL import Image
import pytesseract
from docx import Document
import PyPDF2
from unstructured.partition.auto import partition
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles processing of various document formats."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )
    
    def process_pdf(self, file_path: Path) -> str:
        """Extract text from PDF files."""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return ""
    
    def process_docx(self, file_path: Path) -> str:
        """Extract text from Word documents."""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text.strip()
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path}: {e}")
            return ""
    
    def process_csv(self, file_path: Path) -> str:
        """Convert CSV to readable text format."""
        try:
            df = pd.read_csv(file_path)
            # Convert to structured text
            text = f"Data from {file_path.name}:\n"
            text += df.to_string(index=False)
            return text
        except Exception as e:
            logger.error(f"Error processing CSV {file_path}: {e}")
            return ""
    
    def process_image(self, file_path: Path) -> str:
        """Extract text from images using OCR."""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {e}")
            return ""
    
    def process_text(self, file_path: Path) -> str:
        """Process plain text files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return ""
    
    def process_file(self, file_path: Path) -> str:
        """Route file to appropriate processor based on extension."""
        extension = file_path.suffix.lower()
        
        processors = {
            '.pdf': self.process_pdf,
            '.docx': self.process_docx,
            '.doc': self.process_docx,
            '.csv': self.process_csv,
            '.txt': self.process_text,
            '.md': self.process_text,
            '.jpg': self.process_image,
            '.jpeg': self.process_image,
            '.png': self.process_image,
        }
        
        processor = processors.get(extension)
        if processor:
            logger.info(f"Processing {file_path.name} ({extension})")
            return processor(file_path)
        else:
            logger.warning(f"Unsupported file type: {extension}")
            return ""
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks with metadata."""
        if not text.strip():
            return []
        
        chunks = self.text_splitter.split_text(text)
        return [
            {
                "content": chunk,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "chunk_count": len(chunks)
                }
            }
            for i, chunk in enumerate(chunks)
        ]

class DataIngestionPipeline:
    """Main pipeline for ingesting and processing documents."""
    
    def __init__(self):
        self.processor = DocumentProcessor()
    
    def ingest_documents(self) -> List[Dict[str, Any]]:
        """Process all documents in the raw data directory."""
        all_chunks = []
        
        for file_path in settings.RAW_DATA_DIR.rglob("*"):
            if file_path.is_file():
                # Extract text from file
                text = self.processor.process_file(file_path)
                
                if text:
                    # Create metadata
                    metadata = {
                        "source": str(file_path),
                        "filename": file_path.name,
                        "file_type": file_path.suffix.lower(),
                        "file_size": file_path.stat().st_size,
                    }
                    
                    # Chunk the text
                    chunks = self.processor.chunk_text(text, metadata)
                    all_chunks.extend(chunks)
                    
                    logger.info(f"Processed {file_path.name}: {len(chunks)} chunks")
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def save_processed_data(self, chunks: List[Dict[str, Any]]) -> Path:
        """Save processed chunks to disk."""
        output_file = settings.PROCESSED_DATA_DIR / "processed_chunks.json"
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(chunks)} chunks to {output_file}")
        return output_file

# Example usage
if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    chunks = pipeline.ingest_documents()
    pipeline.save_processed_data(chunks)
