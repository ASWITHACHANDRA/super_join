"""
Cross-document relationship engine.

For each newly extracted fact, retrieves semantically similar facts
from OTHER documents and runs an LLM "judge" call to classify
the relationship as: CORROBORATES | CONTRADICTS | RECONCILABLE | UNRELATED.

Incremental by design: only new_fact → existing_facts pairs are judged,
never existing → existing (no recomputation on new PDF upload).
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .llm_client import llm_client
from .models import Fact, Relationship
from .prompts import JUDGE_SYSTEM, build_judge_prompt
from .vector_store import vector_store

logger = logging.getLogger(__name__)


async def _load_fact_dict(session: AsyncSession, fact_id: str) -> dict | None:
    """Load a Fact from SQLite and return as dict for use in prompts."""
    result = await session.execute(select(Fact).where(Fact.id == fact_id))
    fact = result.scalar_one_or_none()
    if not fact:
        return None
    return {
        "id": fact.id,
        "claim": fact.claim,
        "subject": fact.subject,
        "value": fact.value,
        "time_scope": fact.time_scope,
        "other_scope": fact.other_scope,
        "quote": fact.quote,
        "page_number": fact.page_number,
        "document_id": fact.document_id,
        "document_name": fact.document_name,
        "extraction_notes": fact.extraction_notes,
        "fact_type": fact.fact_type,
    }


async def _relationship_exists(
    session: AsyncSession,
    fact_id_a: str,
    fact_id_b: str,
) -> bool:
    """Check if a relationship between two facts (in either direction) already exists."""
    result = await session.execute(
        select(Relationship).where(
            (
                (Relationship.fact_id_a == fact_id_a) &
                (Relationship.fact_id_b == fact_id_b)
            ) | (
                (Relationship.fact_id_a == fact_id_b) &
                (Relationship.fact_id_b == fact_id_a)
            )
        )
    )
    return result.scalar_one_or_none() is not None


async def judge_pair(
    fact_a: dict,
    fact_b: dict,
) -> dict:
    """
    Call the LLM judge to determine the relationship between two facts.
    Returns a relationship dict.
    """
    prompt = build_judge_prompt(
        fact_a=fact_a,
        fact_b=fact_b,
        doc_a_name=fact_a.get("document_name", "Document A"),
        doc_b_name=fact_b.get("document_name", "Document B"),
    )

    try:
        result = await llm_client.generate_json(
            system_prompt=JUDGE_SYSTEM,
            user_prompt=prompt,
            model=settings.active_judge_model,
            temperature=0.1,
        )
    except Exception as e:
        logger.error(f"Judge call failed for {fact_a['id']} vs {fact_b['id']}: {e}")
        result = {
            "relation": "UNRELATED",
            "explanation": f"Judge call failed: {e}",
            "explanation_basis": ["judge_error"],
        }

    # Validate relation value
    valid_relations = {"CORROBORATES", "CONTRADICTS", "RECONCILABLE", "UNRELATED"}
    relation = result.get("relation", "UNRELATED").upper()
    if relation not in valid_relations:
        relation = "UNRELATED"

    return {
        "id": str(uuid.uuid4()),
        "fact_id_a": fact_a["id"],
        "fact_id_b": fact_b["id"],
        "relation": relation,
        "explanation": result.get("explanation", ""),
        "explanation_basis": result.get("explanation_basis", []),
    }


async def compute_relationships_for_facts(
    new_facts: list[dict],
    session: AsyncSession,
    progress_callback=None,
) -> list[dict]:
    """
    For each new fact, find similar facts from OTHER documents
    and judge relationships.

    Incremental: only new_fact → existing_facts pairs.
    Returns list of relationship dicts ready to be stored.
    """
    all_relationships: list[dict] = []
    total = len(new_facts)
    done = 0

    for fact in new_facts:
        # Find similar facts from other documents
        candidates = await vector_store.search_similar(
            fact=fact,
            exclude_document_id=fact["document_id"],
            top_k=settings.top_k_candidates,
        )

        if not candidates:
            done += 1
            if progress_callback:
                progress_callback(done, total, "relationship")
            continue

        # Load candidate facts from SQLite and judge each pair
        for candidate in candidates:
            cand_id = candidate["id"]

            # Skip if relationship already exists
            if await _relationship_exists(session, fact["id"], cand_id):
                continue

            # Load full candidate fact from DB
            fact_b = await _load_fact_dict(session, cand_id)
            if not fact_b:
                continue

            # Skip UNRELATED (skip judging if similarity is marginal — saves LLM calls)
            # But always judge if similarity > 0.82 (high confidence match)
            if candidate["similarity"] < 0.75:
                # Still record as UNRELATED to avoid re-querying
                rel = {
                    "id": str(uuid.uuid4()),
                    "fact_id_a": fact["id"],
                    "fact_id_b": cand_id,
                    "relation": "UNRELATED",
                    "explanation": f"Similarity score {candidate['similarity']:.3f} below threshold for judging.",
                    "explanation_basis": ["low_similarity"],
                    "similarity_score": candidate["similarity"],
                }
            else:
                rel = await judge_pair(fact, fact_b)
                rel["similarity_score"] = candidate["similarity"]

            all_relationships.append(rel)

        done += 1
        if progress_callback:
            progress_callback(done, total, "relationship")

    logger.info(
        f"Relationship engine: {len(all_relationships)} relationships computed "
        f"for {total} new facts."
    )
    return all_relationships
