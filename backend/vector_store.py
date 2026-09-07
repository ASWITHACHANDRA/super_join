"""
ChromaDB vector store wrapper for FactLens.

Design principles:
- Append-only: adding new documents only adds new vectors, never recomputes existing ones.
- Each fact is stored as a document in ChromaDB with its claim text as the content
  and fact metadata as ChromaDB metadata fields.
- Retrieval returns top-k facts from OTHER documents (cross-document search).
"""
from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import settings
from .llm_client import llm_client

logger = logging.getLogger(__name__)

COLLECTION_NAME = "factlens_facts"


class VectorStore:
    """
    Thin wrapper around ChromaDB for fact storage and retrieval.
    Uses custom embeddings from our LLM client (Gemini/OpenAI).
    """

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # Use custom embedding function backed by our LLM client
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )
        logger.info(
            f"VectorStore initialized. Collection '{COLLECTION_NAME}' "
            f"has {self._collection.count()} facts."
        )

    async def add_facts(self, facts: list[dict]) -> None:
        """
        Embed and add a batch of facts to ChromaDB.
        facts must have: id, claim, document_id, document_name, page_number, fact_type
        """
        if not facts:
            return

        # Build texts to embed: claim + subject + value for richer embedding
        texts = []
        for f in facts:
            parts = [f.get("claim", "")]
            if f.get("subject"):
                parts.append(f"Subject: {f['subject']}")
            if f.get("value"):
                parts.append(f"Value: {f['value']}")
            if f.get("time_scope"):
                parts.append(f"Period: {f['time_scope']}")
            texts.append(" | ".join(parts))

        # Get embeddings in batch
        embeddings = await llm_client.embed(texts)

        # Prepare ChromaDB documents
        ids = [f["id"] for f in facts]
        metadatas = [
            {
                "document_id": f.get("document_id", ""),
                "document_name": f.get("document_name", ""),
                "page_number": f.get("page_number") or 0,
                "fact_type": f.get("fact_type") or "unknown",
                "subject": (f.get("subject") or "")[:500],  # ChromaDB metadata length limit
                "value": (f.get("value") or "")[:500],
                "time_scope": (f.get("time_scope") or "")[:200],
            }
            for f in facts
        ]
        documents = [f.get("claim", "") for f in facts]

        # Batch insert (ChromaDB handles dedup by id)
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(facts)} facts to vector store.")

    async def search_similar(
        self,
        fact: dict,
        exclude_document_id: str,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Find top-k semantically similar facts from documents OTHER than the given one.

        Returns list of dicts with: id, similarity, metadata, document
        """
        k = top_k or settings.top_k_candidates

        # Build query text matching how we embed facts
        parts = [fact.get("claim", "")]
        if fact.get("subject"):
            parts.append(f"Subject: {fact['subject']}")
        if fact.get("value"):
            parts.append(f"Value: {fact['value']}")
        if fact.get("time_scope"):
            parts.append(f"Period: {fact['time_scope']}")
        query_text = " | ".join(parts)

        query_embedding = await llm_client.embed([query_text])

        # Fetch more than k to allow filtering by document
        fetch_n = min(k * 3, max(50, self._collection.count()))
        if fetch_n == 0:
            return []

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=fetch_n,
            include=["distances", "metadatas", "documents"],
        )

        candidates = []
        if not results["ids"] or not results["ids"][0]:
            return candidates

        for i, cand_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            # Skip facts from the same document (cross-document only)
            if meta.get("document_id") == exclude_document_id:
                continue
            # Skip the fact itself
            if cand_id == fact.get("id"):
                continue

            distance = results["distances"][0][i]
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity: similarity = 1 - distance/2
            similarity = 1.0 - distance / 2.0

            if similarity < settings.min_similarity:
                continue

            candidates.append({
                "id": cand_id,
                "similarity": similarity,
                "metadata": meta,
                "document": results["documents"][0][i],
            })

            if len(candidates) >= k:
                break

        return candidates

    def get_fact_count(self) -> int:
        return self._collection.count()

    def fact_exists(self, fact_id: str) -> bool:
        result = self._collection.get(ids=[fact_id])
        return len(result["ids"]) > 0


# Global singleton
vector_store = VectorStore()
