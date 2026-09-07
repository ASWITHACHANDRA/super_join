"""
Fact extraction: runs LLM extraction prompt over each chunk and returns
structured Fact objects ready to be stored.

No document-specific logic. Works on any chunk from any domain.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from .config import settings
from .ingestion import Chunk
from .llm_client import llm_client
from .prompts import EXTRACTION_SYSTEM, build_extraction_prompt

logger = logging.getLogger(__name__)


def _validate_fact(raw: dict, chunk: Chunk, document_name: str) -> dict | None:
    """
    Validate and normalize a raw fact dict from the LLM.
    Returns None if the fact is malformed or not a genuine claim.
    """
    claim = raw.get("claim", "").strip()
    if not claim or len(claim) < 10:
        return None

    quote = raw.get("quote", "").strip()
    confidence = raw.get("confidence", 0.5)

    # Track quote character offsets within the chunk text
    quote_start = raw.get("quote_start")
    quote_end = raw.get("quote_end")

    # Verify/correct char offsets if possible
    if quote and (quote_start is None or quote_end is None):
        idx = chunk.text.find(quote[:50])  # search by first 50 chars
        if idx != -1:
            quote_start = chunk.start_char + idx
            quote_end = quote_start + len(quote)
        else:
            quote_start = chunk.start_char
            quote_end = chunk.end_char

    # Log new fact_type values that haven't been seen before (schema evolution)
    fact_type = raw.get("fact_type", "unknown")

    return {
        "id": str(uuid.uuid4()),
        "document_id": chunk.document_id,
        "document_name": document_name,
        "page_number": chunk.page_number,
        "chunk_id": chunk.chunk_id,
        "claim": claim,
        "subject": raw.get("subject", "").strip() or None,
        "value": raw.get("value", "").strip() or None,
        "time_scope": raw.get("time_scope", "").strip() or None,
        "other_scope": raw.get("other_scope", "").strip() or None,
        "quote": quote or None,
        "quote_start": quote_start,
        "quote_end": quote_end,
        "confidence": float(confidence) if confidence is not None else 0.5,
        "extraction_notes": raw.get("extraction_notes") or None,
        "fact_type": fact_type,
        "metadata_json": {
            k: v for k, v in raw.items()
            if k not in {
                "claim", "subject", "value", "time_scope", "other_scope",
                "quote", "quote_start", "quote_end", "confidence",
                "extraction_notes", "fact_type"
            }
        } or None,
    }


async def extract_facts_from_chunk(
    chunk: Chunk,
    document_name: str,
    semaphore: asyncio.Semaphore,
    seen_fact_types: set[str],
) -> list[dict]:
    """
    Extract facts from a single chunk.
    semaphore limits concurrent LLM calls.
    seen_fact_types is mutated in-place to track schema evolution.
    """
    async with semaphore:
        prompt = build_extraction_prompt(chunk.text, chunk.page_number, document_name)
        try:
            raw_facts = await llm_client.generate_json(
                system_prompt=EXTRACTION_SYSTEM,
                user_prompt=prompt,
                model=settings.active_extraction_model,
                temperature=0.1,
                expect_array=True,
            )
        except Exception as e:
            logger.error(f"Extraction failed for chunk {chunk.chunk_id}: {e}")
            return []

        if not isinstance(raw_facts, list):
            logger.warning(f"Non-list response from extraction: {type(raw_facts)}")
            return []

        validated: list[dict] = []
        for raw in raw_facts:
            if not isinstance(raw, dict):
                continue
            fact = _validate_fact(raw, chunk, document_name)
            if fact is None:
                continue

            # Schema evolution tracking
            ft = fact.get("fact_type", "unknown")
            if ft and ft not in seen_fact_types:
                logger.info(f"[SCHEMA EVOLUTION] New fact_type encountered: '{ft}'")
                seen_fact_types.add(ft)

            validated.append(fact)

        logger.info(f"Chunk {chunk.chunk_id} (page {chunk.page_number}): {len(validated)} facts extracted")
        return validated


async def extract_facts_from_chunks(
    chunks: list[Chunk],
    document_name: str,
    progress_callback=None,
) -> list[dict]:
    """
    Extract facts from all chunks of a document in parallel (bounded concurrency).

    progress_callback(done, total) is called after each chunk completes.
    Returns flat list of validated fact dicts.
    """
    semaphore = asyncio.Semaphore(settings.max_concurrent_extractions)
    seen_fact_types: set[str] = set()
    all_facts: list[dict] = []
    total = len(chunks)
    done = 0

    # Process in parallel with semaphore
    tasks = [
        extract_facts_from_chunk(chunk, document_name, semaphore, seen_fact_types)
        for chunk in chunks
    ]

    for coro in asyncio.as_completed(tasks):
        chunk_facts = await coro
        all_facts.extend(chunk_facts)
        done += 1
        if progress_callback:
            progress_callback(done, total)

    logger.info(
        f"Document '{document_name}': {len(all_facts)} facts from {total} chunks. "
        f"Fact types seen: {sorted(seen_fact_types)}"
    )
    return all_facts
