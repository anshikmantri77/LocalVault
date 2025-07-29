"""Utility functions for the chatbot system."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib
import time
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages configuration and settings."""
    
    @staticmethod
    def load_config(config_path: Path) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: Path) -> bool:
        """Save configuration to JSON file."""
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            return False

class FileHasher:
    """Handles file hashing for change detection."""
    
    @staticmethod
    def get_file_hash(file_path: Path) -> str:
        """Get MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return ""
    
    @staticmethod
    def has_file_changed(file_path: Path, stored_hash: str) -> bool:
        """Check if file has changed since last hash."""
        current_hash = FileHasher.get_file_hash(file_path)
        return current_hash != stored_hash

class PerformanceMonitor:
    """Monitors system performance and usage."""
    
    def __init__(self):
        self.metrics = []
    
    def log_metric(self, operation: str, duration: float, **kwargs):
        """Log a performance metric."""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration": duration,
            **kwargs
        }
        self.metrics.append(metric)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.metrics:
            return {}
        
        df = pd.DataFrame(self.metrics)
        return {
            "total_operations": len(self.metrics),
            "avg_duration": df["duration"].mean(),
            "max_duration": df["duration"].max(),
            "min_duration": df["duration"].min(),
            "operations_by_type": df["operation"].value_counts().to_dict()
        }
    
    def clear_metrics(self):
        """Clear stored metrics."""
        self.metrics.clear()

class DocumentAnalyzer:
    """Analyzes document content for insights."""
    
    @staticmethod
    def extract_keywords(text: str, top_k: int = 10) -> List[str]:
        """Extract top keywords from text."""
        # Simple keyword extraction (can be enhanced with NLP libraries)
        import re
        from collections import Counter
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'this',
            'that', 'these', 'those', 'will', 'would', 'could', 'should',
            'may', 'might', 'must', 'can', 'have', 'has', 'had', 'are', 'is',
            'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'done', 'doing'
        }
        
        # Filter stop words and count
        filtered_words = [word for word in words if word not in stop_words]
        word_counts = Counter(filtered_words)
        
        return [word for word, count in word_counts.most_common(top_k)]
    
    @staticmethod
    def analyze_document_types(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze distribution of document types."""
        type_counts = {}
        total_size = 0
        
        for chunk in chunks:
            file_type = chunk["metadata"].get("file_type", "unknown")
            file_size = chunk["metadata"].get("file_size", 0)
            
            if file_type not in type_counts:
                type_counts[file_type] = {"count": 0, "size": 0}
            
            type_counts[file_type]["count"] += 1
            type_counts[file_type]["size"] += file_size
            total_size += file_size
        
        return {
            "type_distribution": type_counts,
            "total_size": total_size,
            "total_chunks": len(chunks)
        }

# Decorators for timing and logging
def timed_operation(operation_name: str):
    """Decorator to time operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"{operation_name} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"{operation_name} failed after {duration:.2f}s: {e}")
                raise
        return wrapper
    return decorator

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry operations on failure."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Operation failed after {max_retries} attempts: {e}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator
