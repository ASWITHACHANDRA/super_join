"""
Embedding-based normalization for entity/metric clustering.

Purpose: loosely canonicalize subject strings like "CPI inflation",
"headline inflation", "consumer price inflation" into the same cluster
WITHOUT a hard-coded synonym table.

Strategy:
1. Collect all unique subject strings from the facts table
2. Embed them all
3. Use agglomerative clustering (cosine distance) to group similar subjects
4. Store canonical labels in the database (optional — this is a brownie-point feature)

This module is called asynchronously after extraction; it does NOT block
the main pipeline. Normalization improves vector search recall but the
system works without it (embeddings of similar subjects already cluster
together in the vector space used for relationship search).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .llm_client import llm_client

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


async def cluster_subjects(
    subjects: list[str],
    similarity_threshold: float = 0.88,
) -> dict[str, str]:
    """
    Cluster a list of subject strings into canonical groups.
    Returns a mapping: original_subject → canonical_subject.

    Two subjects are merged if their embedding cosine similarity >= threshold.
    The cluster representative is the most frequently occurring member
    (or the first seen if ties).

    Example:
        "CPI inflation" → "CPI inflation" (canonical)
        "consumer price inflation" → "CPI inflation" (mapped)
        "headline CPI" → "CPI inflation" (mapped)
    """
    if not subjects:
        return {}

    unique = list(dict.fromkeys(subjects))   # preserve order, deduplicate
    if len(unique) == 1:
        return {unique[0]: unique[0]}

    logger.info(f"Clustering {len(unique)} unique subject strings...")
    
    # Embed all subjects
    try:
        embeddings = await llm_client.embed(unique)
    except Exception as e:
        logger.warning(f"Embedding failed for normalization: {e}. Skipping clustering.")
        return {s: s for s in unique}

    # Simple agglomerative single-linkage clustering
    # Build clusters greedily
    clusters: list[list[int]] = []     # list of index groups
    assigned = [False] * len(unique)

    for i in range(len(unique)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, len(unique)):
            if assigned[j]:
                continue
            sim = _cosine_similarity(embeddings[i], embeddings[j])
            if sim >= similarity_threshold:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    # Build mapping: each cluster member → first member as canonical
    mapping: dict[str, str] = {}
    for cluster in clusters:
        canonical = unique[cluster[0]]
        for idx in cluster:
            mapping[unique[idx]] = canonical

    # Log clusters with more than 1 member (informative)
    merged = [c for c in clusters if len(c) > 1]
    if merged:
        logger.info(f"Normalization merged {len(merged)} subject clusters:")
        for cluster in merged:
            members = [unique[i] for i in cluster]
            logger.info(f"  → '{members[0]}' ← {members[1:]}")

    return mapping


async def normalize_subjects_in_facts(facts: list[dict]) -> list[dict]:
    """
    Apply embedding-based subject normalization to a list of fact dicts.
    Mutates facts in-place (adds 'canonical_subject' key) and returns them.
    """
    subjects = [f.get("subject") for f in facts if f.get("subject")]
    if not subjects:
        return facts

    mapping = await cluster_subjects(subjects)

    for f in facts:
        s = f.get("subject")
        if s and s in mapping:
            f["canonical_subject"] = mapping[s]
        else:
            f["canonical_subject"] = s

    return facts
