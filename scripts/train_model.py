#!/usr/bin/env python3
"""Script to fine-tune the LLM model."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.fine_tuning import DatasetGenerator, FineTuner
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the fine-tuning pipeline."""
    print("🎯 Starting model fine-tuning...")
    
    # Step 1: Generate dataset
    print("📊 Generating training dataset...")
    dataset_generator = DatasetGenerator()
    dataset = dataset_generator.create_training_dataset()
    
    if len(dataset) == 0:
        print("⚠️  No training data available. Run data ingestion first.")
        return
    
    print(f"   - Generated {len(dataset)} training examples")
    
    # Step 2: Fine-tune model
    print("🔧 Starting fine-tuning (this may take a while)...")
    fine_tuner = FineTuner()
    model_path = fine_tuner.fine_tune(dataset)
    
    # Step 3: Deploy to Ollama
    print("🚀 Deploying model to Ollama...")
    success = fine_tuner.deploy_to_ollama(model_path)
    
    if success:
        print("✅ Fine-tuning complete! Model deployed to Ollama.")
    else:
        print("⚠️  Fine-tuning completed but deployment to Ollama failed.")
        print(f"   Model saved at: {model_path}")

if __name__ == "__main__":
    main()
