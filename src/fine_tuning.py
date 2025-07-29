"""Fine-tuning pipeline for personalizing the LLM."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import ollama
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetGenerator:
    """Generates training datasets from processed chunks."""
    
    def __init__(self):
        self.chunks = self.load_chunks()
    
    def load_chunks(self) -> List[Dict[str, Any]]:
        """Load processed chunks from disk."""
        chunks_file = settings.PROCESSED_DATA_DIR / "processed_chunks.json"
        if chunks_file.exists():
            with open(chunks_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def generate_qa_pairs(self) -> List[Dict[str, str]]:
        """Generate question-answer pairs from chunks."""
        qa_pairs = []
        
        for chunk in self.chunks:
            content = chunk["content"]
            filename = chunk["metadata"]["filename"]
            
            # Generate contextual questions based on content
            questions = self.generate_questions_for_content(content, filename)
            
            for question in questions:
                qa_pairs.append({
                    "instruction": question,
                    "input": "",
                    "output": content,
                    "source": filename
                })
        
        logger.info(f"Generated {len(qa_pairs)} Q&A pairs")
        return qa_pairs
    
    def generate_questions_for_content(self, content: str, filename: str) -> List[str]:
        """Generate relevant questions for a piece of content."""
        questions = []
        
        # Generic questions
        questions.extend([
            f"What information do you have about {filename}?",
            f"Tell me about the content in {filename}",
            "What does this document contain?",
            "Summarize this information"
        ])
        
        # Content-specific questions based on keywords
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['project', 'task', 'deadline']):
            questions.append("What projects or tasks are mentioned?")
        
        if any(word in content_lower for word in ['meeting', 'discussion', 'agenda']):
            questions.append("What meetings or discussions are referenced?")
        
        if any(word in content_lower for word in ['data', 'analysis', 'results']):
            questions.append("What data or analysis is presented?")
        
        return questions[:3]  # Limit to 3 questions per chunk
    
    def create_training_dataset(self) -> Dataset:
        """Create a Hugging Face dataset for fine-tuning."""
        qa_pairs = self.generate_qa_pairs()
        
        # Format for instruction tuning
        formatted_data = []
        for pair in qa_pairs:
            formatted_text = f"### Instruction:\n{pair['instruction']}\n\n### Response:\n{pair['output']}"
            formatted_data.append({"text": formatted_text})
        
        return Dataset.from_list(formatted_data)

class FineTuner:
    """Handles fine-tuning of the LLM using LoRA."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # We'll use a compatible model for local fine-tuning
        self.model_name = "microsoft/DialoGPT-medium"  # Smaller model for local training
        self.tokenizer = None
        self.model = None
    
    def setup_model(self):
        """Initialize model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=settings.LORA_RANK,
            lora_alpha=settings.LORA_ALPHA,
            lora_dropout=settings.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["c_attn", "c_proj"]  # DialoGPT specific
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def tokenize_function(self, examples):
        """Tokenize examples for training."""
        outputs = self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        outputs["labels"] = outputs["input_ids"].clone()
        return outputs
    
    def fine_tune(self, dataset: Dataset) -> str:
        """Fine-tune the model on the dataset."""
        if not self.model:
            self.setup_model()
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(settings.MODELS_DIR / "fine_tuned"),
            per_device_train_batch_size=settings.BATCH_SIZE,
            num_train_epochs=settings.NUM_EPOCHS,
            learning_rate=settings.LEARNING_RATE,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Save model
        output_dir = settings.MODELS_DIR / "fine_tuned_lora"
        trainer.save_model(str(output_dir))
        self.tokenizer.save_pretrained(str(output_dir))
        
        logger.info(f"Fine-tuning completed. Model saved to {output_dir}")
        return str(output_dir)
    
    def create_ollama_modelfile(self, model_path: str) -> str:
        """Create Ollama Modelfile for deployment."""
        modelfile_content = f"""
FROM {settings.BASE_MODEL}

# Custom instructions for your personal assistant
SYSTEM You are a helpful personal assistant with access to the user's documents and information. Always provide accurate, relevant responses based on the available context.

# Fine-tuned adapter
ADAPTER {model_path}

# Parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
"""
        
        modelfile_path = settings.MODELS_DIR / "Modelfile"
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        return str(modelfile_path)
    
    def deploy_to_ollama(self, model_path: str) -> bool:
        """Deploy fine-tuned model to Ollama."""
        try:
            # Create Modelfile
            modelfile_path = self.create_ollama_modelfile(model_path)
            
            # Build custom model in Ollama
            logger.info(f"Creating Ollama model: {settings.FINE_TUNED_MODEL}")
            
            # Note: This is a simplified version. In practice, you'd need to
            # convert the fine-tuned weights to Ollama format
            ollama.create(
                model=settings.FINE_TUNED_MODEL,
                modelfile=modelfile_path
            )
            
            logger.info("Model successfully deployed to Ollama")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying to Ollama: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Generate dataset
    dataset_generator = DatasetGenerator()
    dataset = dataset_generator.create_training_dataset()
    
    # Fine-tune model
    fine_tuner = FineTuner()
    model_path = fine_tuner.fine_tune(dataset)
    
    # Deploy to Ollama
    fine_tuner.deploy_to_ollama(model_path)
