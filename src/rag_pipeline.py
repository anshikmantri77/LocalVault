"""RAG pipeline for combining retrieval with generation."""

import logging
from typing import List, Dict, Any, Optional
import ollama
from src.embeddings import EmbeddingManager
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Implements Retrieval-Augmented Generation pipeline."""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        
        # Verify model availability
        self.model_name = self._get_available_model()
        logger.info(f"Using model: {self.model_name}")
    
    def _get_available_model(self) -> str:
        """Get the best available model."""
        try:
            models = self.ollama_client.list()
            model_names = [model['name'] for model in models['models']]
            
            # Prefer fine-tuned model if available
            if settings.FINE_TUNED_MODEL in model_names:
                return settings.FINE_TUNED_MODEL
            elif settings.BASE_MODEL in model_names:
                return settings.BASE_MODEL
            else:
                # Use the first available model
                return model_names[0] if model_names else None
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return settings.BASE_MODEL
    
    def retrieve_context(
        self, 
        query: str, 
        k: int = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context for the query."""
        k = k or settings.TOP_K_RESULTS
        
        results = self.embedding_manager.similarity_search(
            query=query,
            k=k,
            filter_metadata=filter_metadata
        )
        
        logger.info(f"Retrieved {len(results)} context chunks for query")
        return results
    
    def build_rag_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Build a prompt that combines query with retrieved context."""
        
        # Build context section
        context_text = ""
        sources = []
        
        for i, chunk in enumerate(context_chunks, 1):
            content = chunk["content"]
            source = chunk["metadata"]["filename"]
            
            context_text += f"[Context {i}] From {source}:\n{content}\n\n"
            sources.append(source)
        
        # Create the final prompt
        prompt = f"""You are a helpful personal assistant with access to the user's documents. Use the provided context to answer the question accurately and helpfully.

Context Information:
{context_text}

Question: {query}

Instructions:
- Use the context information to provide an accurate answer
- If the context doesn't contain relevant information, say so
- Cite specific sources when referencing information
- Be conversational and helpful

Answer:"""
        
        return prompt
    
    def generate_response(self, prompt: str) -> Dict[str, Any]:
        """Generate response using Ollama."""
        try:
            response = self.ollama_client.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                }
            )
            
            return {
                "content": response['message']['content'],
                "model": self.model_name,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "content": "Sorry, I encountered an error generating a response.",
                "model": self.model_name,
                "success": False,
                "error": str(e)
            }
    
    def chat(
        self, 
        query: str, 
        include_sources: bool = True,
        k: int = None
    ) -> Dict[str, Any]:
        """Main chat function that combines retrieval and generation."""
        
        # Step 1: Retrieve relevant context
        context_chunks = self.retrieve_context(query, k)
        
        # Step 2: Build RAG prompt
        rag_prompt = self.build_rag_prompt(query, context_chunks)
        
        # Step 3: Generate response
        response = self.generate_response(rag_prompt)
        
        # Step 4: Compile final result
        result = {
            "query": query,
            "response": response["content"],
            "model": response["model"],
            "success": response["success"]
        }
        
        if include_sources:
            result["sources"] = [
                {
                    "filename": chunk["metadata"]["filename"],
                    "content_preview": chunk["content"][:200] + "...",
                    "similarity": chunk["similarity"]
                }
                for chunk in context_chunks
            ]
            result["context_used"] = len(context_chunks)
        
        if not response["success"]:
            result["error"] = response.get("error")
        
        return result

# Conversation memory for chat history
class ConversationMemory:
    """Manages conversation history and context."""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []
    
    def add_exchange(self, query: str, response: str):
        """Add a query-response exchange to history."""
        self.history.append({
            "query": query,
            "response": response,
            "timestamp": self._get_timestamp()
        })
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context(self, last_n: int = 3) -> str:
        """Get recent conversation context."""
        if not self.history:
            return ""
        
        recent_history = self.history[-last_n:]
        context = "Recent conversation:\n"
        
        for exchange in recent_history:
            context += f"Q: {exchange['query']}\n"
            context += f"A: {exchange['response'][:100]}...\n\n"
        
        return context
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

class EnhancedRAGPipeline(RAGPipeline):
    """Enhanced RAG pipeline with conversation memory."""
    
    def __init__(self):
        super().__init__()
        self.memory = ConversationMemory()
    
    def chat_with_memory(
        self, 
        query: str, 
        use_conversation_context: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Chat with conversation memory."""
        
        # Add conversation context to query if requested
        if use_conversation_context and self.memory.history:
            conversation_context = self.memory.get_context()
            enhanced_query = f"{conversation_context}\nCurrent question: {query}"
        else:
            enhanced_query = query
        
        # Get response using parent chat method
        result = self.chat(enhanced_query, **kwargs)
        
        # Add to memory
        if result["success"]:
            self.memory.add_exchange(query, result["response"])
        
        return result

# Example usage
if __name__ == "__main__":
    # Test RAG pipeline
    rag = EnhancedRAGPipeline()
    
    # Test queries
    test_queries = [
        "What documents do I have?",
        "Summarize the main topics in my files",
        "What projects am I working on?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = rag.chat_with_memory(query)
        
        if result["success"]:
            print(f"Response: {result['response']}")
            print(f"Sources used: {len(result.get('sources', []))}")
        else:
            print(f"Error: {result.get('error')}")
