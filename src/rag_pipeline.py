"""
RAG pipeline — clean, production-grade implementation.

Fixes applied:
  1. Hallucination Guard: removed broken regex entity-checker that was
     cutting valid answers. Replaced with a strict grounded prompt.
  2. Context Leakage: similarity threshold prevents weak/irrelevant chunks.
  3. Score Glitch: threshold (0.30) gates retrieval cleanly.
  4. Short Internship Answers: k=8 for personal queries pulls enough chunks.
  5. Repetitive Looping: ConversationMemory dedup prevents re-summarising.
  6. Hardcoded Patch Fixes: all keyword if/else patches removed; clean QueryRouter.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional

from src.embeddings import EmbeddingManager
from src.ollama_client import generate
from config.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

RAG_PROMPT_TEMPLATE = """\
You are a precise Document Assistant. Answer the QUESTION using ONLY the information \
in the CONTEXT below. Do not add, infer, or invent any information not explicitly stated.

CONTEXT:
{context}

QUESTION: {query}

RULES:
- If the answer is clearly present in the CONTEXT, give a complete and well-structured answer.
- Use bullet points when listing multiple items (skills, internships, projects, etc.).
- If the CONTEXT does not contain enough information, respond ONLY with:
  "I could not find that information in the provided documents."
- Never guess, never use general knowledge.

ANSWER:"""

CONVERSATIONAL_PROMPT_TEMPLATE = """\
You are a helpful assistant. The user is asking a general question.

{history}

QUESTION: {query}

ANSWER:"""


# ---------------------------------------------------------------------------
# Query Router  (replaces all hardcoded keyword patches)
# ---------------------------------------------------------------------------

class QueryRouter:
    """Classifies a query so the pipeline applies the right retrieval settings."""

    PERSONAL_KEYWORDS = {
        "name", "college", "university", "degree", "education", "studied",
        "internship", "intern", "experience", "work", "job", "role", "position",
        "project", "skill", "certificate", "certification", "gpa", "cgpa",
        "resume", "cv", "background", "qualification", "nlp", "ml", "ai",
        "python", "java", "django", "flask", "deep learning", "machine learning",
        "where did", "when did", "who are", "what did", "tell me about",
    }

    FINANCIAL_KEYWORDS = {
        "revenue", "profit", "ebitda", "sales", "cagr", "irr", "npv", "margin",
        "projection", "forecast", "valuation", "fy", "financial", "table",
        "appendix", "2024", "2025", "2026", "2027", "2028", "cim", "investment",
        "deal", "business", "market", "growth",
    }

    @classmethod
    def classify(cls, query: str) -> str:
        """Return 'personal', 'financial', or 'general'."""
        q = query.lower()
        if any(kw in q for kw in cls.PERSONAL_KEYWORDS):
            return "personal"
        if any(kw in q for kw in cls.FINANCIAL_KEYWORDS):
            return "financial"
        return "general"


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Clean Retrieval-Augmented Generation pipeline backed by local Ollama."""

    SIMILARITY_THRESHOLD = 0.30  # chunks below this score are discarded

    def __init__(
        self,
        model: Optional[str] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
    ):
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.model = model or settings.OLLAMA_MODEL
        logger.info("RAGPipeline ready. Model: %s", self.model)

    def set_model(self, model_key: str):
        self.model = model_key
        logger.info("Model switched to: %s", self.model)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_context(
        self,
        query: str,
        k: int = 8,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        raw = self.embedding_manager.similarity_search(
            query=query, k=k, filter_metadata=filter_metadata
        )
        # Apply similarity threshold — fixes Issue #3
        filtered = [c for c in raw if c.get("similarity", 0) >= self.SIMILARITY_THRESHOLD]
        # Deduplicate by content prefix to avoid repeated blocks — fixes Issue #5
        seen, unique = set(), []
        for c in filtered:
            key = c["content"][:100]
            if key not in seen:
                seen.add(key)
                unique.append(c)
        logger.info(
            "Retrieval: %d raw → %d after threshold+dedup (threshold=%.2f)",
            len(raw), len(unique), self.SIMILARITY_THRESHOLD,
        )
        return unique

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_rag_prompt(
        self, query: str, chunks: List[Dict[str, Any]], history: str = ""
    ) -> str:
        context_text = ""
        for i, chunk in enumerate(chunks, 1):
            source = chunk["metadata"].get("filename", "unknown")
            context_text += f"[{i}] (source: {source})\n{chunk['content']}\n\n"
        prompt = RAG_PROMPT_TEMPLATE.format(context=context_text, query=query)
        if history:
            prompt = f"CONVERSATION HISTORY:\n{history}\n\n{prompt}"
        return prompt

    def build_general_prompt(self, query: str, history: str = "") -> str:
        return CONVERSATIONAL_PROMPT_TEMPLATE.format(
            history=f"CONVERSATION HISTORY:\n{history}" if history else "",
            query=query,
        )

    # ------------------------------------------------------------------
    # Main chat
    # ------------------------------------------------------------------

    async def chat(
        self,
        query: str,
        include_sources: bool = True,
        k: int = None,
        temperature: float = 0.1,
        history: str = "",
    ) -> Dict[str, Any]:
        """Run one RAG turn."""

        # Route the query — replaces all hardcoded keyword patches (Issue #6)
        query_type = QueryRouter.classify(query)
        logger.info("Query type: %s | Query: %s", query_type, query)

        # Adaptive k by query type — fixes Issue #4 (short answers)
        if k is None:
            if query_type == "personal":
                k = 8
            elif query_type == "financial":
                k = 12
            else:
                k = 5

        # Retrieve relevant chunks
        chunks = self.retrieve_context(query, k=k)

        if not chunks:
            return {
                "query": query,
                "response": (
                    "I could not find any relevant information in your documents "
                    "for this question. Please make sure the correct files are "
                    "uploaded and indexed."
                ),
                "model": self.model,
                "success": True,
                "sources": [],
                "context_used": 0,
            }

        filenames = {c["metadata"].get("filename", "?") for c in chunks}
        logger.info("Context from %d document(s): %s", len(filenames), filenames)

        # Build prompt and generate
        prompt = self.build_rag_prompt(query, chunks, history)
        result = await generate(prompt, model=self.model, temperature=temperature)
        response = result["content"].strip()

        # Clean fallback for empty model output
        if not response:
            response = (
                "I found relevant document sections but the model returned an empty "
                "response. Try re-phrasing your question or checking Ollama is running."
            )

        output = {
            "query": query,
            "response": response,
            "model": self.model,
            "success": result["success"],
            "sources": [],
            "context_used": len(chunks),
        }

        if include_sources:
            output["sources"] = [
                {
                    "filename": c["metadata"].get("filename", "unknown"),
                    "content_preview": c["content"][:200] + "…",
                    "similarity": round(c.get("similarity", 0.0), 3),
                }
                for c in chunks
            ]

        return output


# ---------------------------------------------------------------------------
# Conversation Memory  (fixes Issue #5 — repetitive looping)
# ---------------------------------------------------------------------------

class ConversationMemory:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.history: List[Dict[str, Any]] = []
        self._seen_queries: set = set()

    def add(self, query: str, response: str):
        from datetime import datetime
        key = query.strip().lower()
        if key in self._seen_queries:
            return  # exact duplicate — skip
        self._seen_queries.add(key)
        self.history.append(
            {"query": query, "response": response, "timestamp": datetime.now().isoformat()}
        )
        if len(self.history) > self.max_history:
            oldest_key = self.history[0]["query"].strip().lower()
            self._seen_queries.discard(oldest_key)
            self.history = self.history[-self.max_history:]

    def get_context(self, last_n: int = 3) -> str:
        if not self.history:
            return ""
        recent = self.history[-last_n:]
        lines = ["Recent conversation:"]
        for ex in recent:
            lines.append(f"Q: {ex['query']}")
            ans = ex["response"][:200].rstrip()
            lines.append(f"A: {ans}{'…' if len(ex['response']) > 200 else ''}")
            lines.append("")
        return "\n".join(lines)

    def clear(self):
        self.history.clear()
        self._seen_queries.clear()


# ---------------------------------------------------------------------------
# Enhanced pipeline (with memory)
# ---------------------------------------------------------------------------

class EnhancedRAGPipeline(RAGPipeline):
    """RAG pipeline with multi-turn conversation memory."""

    def __init__(
        self,
        model: Optional[str] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
    ):
        super().__init__(model, embedding_manager)
        self.memory = ConversationMemory()

    async def chat_with_memory(
        self, query: str, use_conversation_context: bool = True, **kwargs
    ) -> Dict[str, Any]:
        history = ""
        if use_conversation_context and self.memory.history:
            history = self.memory.get_context()

        result = await self.chat(query, history=history, **kwargs)

        if result["success"]:
            self.memory.add(query, result["response"])
        return result
