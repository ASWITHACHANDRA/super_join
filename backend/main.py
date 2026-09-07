"""
FactLens FastAPI application.

Routes:
  POST /documents                    — upload PDF, trigger async pipeline
  GET  /documents/:id/status         — processing status
  GET  /documents                    — list all documents
  GET  /facts                        — list facts (filterable by document_id, fact_type)
  GET  /facts/:id                    — single fact with full evidence
  GET  /facts/:id/relationships      — related facts with relation type + explanation
  GET  /relationships                — browse all relationships (filterable by type)
  GET  /                             — serves frontend HTML
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import aiofiles
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .config import settings
from .database import AsyncSessionLocal, get_session, init_db
from .extraction import extract_facts_from_chunks
from .ingestion import ingest_pdf
from .models import Document, Fact, Relationship
from .relationship import compute_relationships_for_facts
from .vector_store import vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FactLens",
    description="Cross-document fact extraction and comparison for PDFs",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    await init_db()
    logger.info("FactLens database initialized.")


# ── Frontend ──────────────────────────────────────────────────────────────────

FRONTEND_PATH = Path(__file__).parent.parent / "frontend" / "index.html"


@app.get("/", include_in_schema=False)
async def serve_ui():
    if FRONTEND_PATH.exists():
        return FileResponse(str(FRONTEND_PATH), media_type="text/html")
    return HTMLResponse("<h1>FactLens API</h1><p>See <a href='/docs'>API docs</a></p>")


# ── Background pipeline ────────────────────────────────────────────────────────

async def run_pipeline(document_id: str, filepath: str, document_name: str) -> None:
    """
    Full async pipeline for a single uploaded PDF:
    1. Ingest → pages + chunks
    2. Extract facts (LLM, parallel)
    3. Store facts in SQLite
    4. Embed facts in ChromaDB
    5. Compute relationships (LLM judge)
    6. Store relationships in SQLite
    """
    async with AsyncSessionLocal() as session:
        # Update status: processing
        doc_result = await session.execute(select(Document).where(Document.id == document_id))
        doc = doc_result.scalar_one_or_none()
        if not doc:
            logger.error(f"Document {document_id} not found in DB")
            return
        doc.status = "processing"
        await session.commit()

        try:
            # ── Step 1: Ingest ─────────────────────────────────────────────
            logger.info(f"[{document_name}] Ingesting PDF...")
            pages, chunks = ingest_pdf(filepath, document_id)

            doc.page_count = len(pages)
            doc.chunk_count = len(chunks)
            await session.commit()

            logger.info(f"[{document_name}] {len(pages)} pages, {len(chunks)} chunks")

            # ── Step 2: Extract facts ──────────────────────────────────────
            logger.info(f"[{document_name}] Extracting facts from {len(chunks)} chunks...")
            raw_facts = await extract_facts_from_chunks(chunks, document_name)

            logger.info(f"[{document_name}] {len(raw_facts)} facts extracted")

            # ── Step 3: Store facts in SQLite ──────────────────────────────
            fact_objects = []
            for f in raw_facts:
                fact_obj = Fact(
                    id=f["id"],
                    document_id=f["document_id"],
                    document_name=f["document_name"],
                    page_number=f.get("page_number"),
                    chunk_id=f.get("chunk_id"),
                    claim=f["claim"],
                    subject=f.get("subject"),
                    value=f.get("value"),
                    time_scope=f.get("time_scope"),
                    other_scope=f.get("other_scope"),
                    quote=f.get("quote"),
                    quote_start=f.get("quote_start"),
                    quote_end=f.get("quote_end"),
                    confidence=f.get("confidence"),
                    extraction_notes=f.get("extraction_notes"),
                    fact_type=f.get("fact_type"),
                    metadata_json=f.get("metadata_json"),
                )
                session.add(fact_obj)
                fact_objects.append(fact_obj)

            doc.fact_count = len(raw_facts)
            await session.commit()

            # ── Step 4: Embed in ChromaDB ──────────────────────────────────
            logger.info(f"[{document_name}] Embedding {len(raw_facts)} facts...")
            await vector_store.add_facts(raw_facts)

            # ── Step 5: Compute relationships ──────────────────────────────
            logger.info(f"[{document_name}] Computing cross-document relationships...")
            relationships = await compute_relationships_for_facts(raw_facts, session)

            # ── Step 6: Store relationships ────────────────────────────────
            for rel in relationships:
                rel_obj = Relationship(
                    id=rel["id"],
                    fact_id_a=rel["fact_id_a"],
                    fact_id_b=rel["fact_id_b"],
                    relation=rel["relation"],
                    explanation=rel.get("explanation"),
                    explanation_basis=rel.get("explanation_basis"),
                    similarity_score=rel.get("similarity_score"),
                )
                session.add(rel_obj)

            await session.commit()

            doc.status = "done"
            await session.commit()

            logger.info(
                f"[{document_name}] Pipeline complete. "
                f"{len(raw_facts)} facts, {len(relationships)} relationships."
            )

        except Exception as e:
            logger.exception(f"[{document_name}] Pipeline error: {e}")
            doc.status = "error"
            doc.error_message = str(e)
            await session.commit()


# ── Document endpoints ────────────────────────────────────────────────────────

@app.post("/documents", summary="Upload a PDF and trigger processing")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: Optional[str] = Query(None, description="Display name for the document"),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    document_id = str(uuid.uuid4())
    display_name = name or Path(file.filename).stem

    # Save uploaded file
    upload_dir = Path(settings.pdf_upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_filename = f"{document_id}_{file.filename}"
    filepath = upload_dir / safe_filename

    async with aiofiles.open(filepath, "wb") as f:
        content = await file.read()
        await f.write(content)

    # Create document record
    async with AsyncSessionLocal() as session:
        doc = Document(
            id=document_id,
            name=display_name,
            filename=file.filename,
            filepath=str(filepath),
            status="pending",
        )
        session.add(doc)
        await session.commit()

    # Kick off async pipeline
    background_tasks.add_task(run_pipeline, document_id, str(filepath), display_name)

    return {
        "document_id": document_id,
        "name": display_name,
        "status": "pending",
        "message": "Processing started. Poll /documents/{id}/status for updates.",
    }


@app.get("/documents", summary="List all documents")
async def list_documents(session: AsyncSession = Depends(get_session)):
    result = await session.execute(
        select(Document).order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return [
        {
            "id": d.id,
            "name": d.name,
            "filename": d.filename,
            "status": d.status,
            "page_count": d.page_count,
            "chunk_count": d.chunk_count,
            "fact_count": d.fact_count,
            "error_message": d.error_message,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in docs
    ]


@app.get("/documents/{document_id}/status", summary="Get document processing status")
async def get_document_status(
    document_id: str,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "id": doc.id,
        "name": doc.name,
        "status": doc.status,
        "page_count": doc.page_count,
        "chunk_count": doc.chunk_count,
        "fact_count": doc.fact_count,
        "error_message": doc.error_message,
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
    }


# ── Fact endpoints ────────────────────────────────────────────────────────────

@app.get("/facts", summary="List facts with optional filtering")
async def list_facts(
    document_id: Optional[str] = Query(None),
    fact_type: Optional[str] = Query(None),
    subject: Optional[str] = Query(None, description="Partial subject match"),
    limit: int = Query(50, le=500),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    query = select(Fact).order_by(Fact.created_at.desc())

    if document_id:
        query = query.where(Fact.document_id == document_id)
    if fact_type:
        query = query.where(Fact.fact_type == fact_type)
    if subject:
        query = query.where(Fact.subject.ilike(f"%{subject}%"))

    query = query.limit(limit).offset(offset)
    result = await session.execute(query)
    facts = result.scalars().all()

    return [_fact_to_dict(f) for f in facts]


@app.get("/facts/{fact_id}", summary="Get a single fact with full evidence")
async def get_fact(
    fact_id: str,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(Fact).where(Fact.id == fact_id))
    fact = result.scalar_one_or_none()
    if not fact:
        raise HTTPException(status_code=404, detail="Fact not found")
    return _fact_to_dict(fact, full=True)


@app.get("/facts/{fact_id}/relationships", summary="Get relationships for a fact")
async def get_fact_relationships(
    fact_id: str,
    session: AsyncSession = Depends(get_session),
):
    # Check fact exists
    fact_result = await session.execute(select(Fact).where(Fact.id == fact_id))
    fact = fact_result.scalar_one_or_none()
    if not fact:
        raise HTTPException(status_code=404, detail="Fact not found")

    # Get all relationships involving this fact (as either party)
    result = await session.execute(
        select(Relationship).where(
            (Relationship.fact_id_a == fact_id) | (Relationship.fact_id_b == fact_id)
        ).order_by(Relationship.relation)
    )
    rels = result.scalars().all()

    output = []
    for rel in rels:
        # Load the "other" fact
        other_id = rel.fact_id_b if rel.fact_id_a == fact_id else rel.fact_id_a
        other_result = await session.execute(select(Fact).where(Fact.id == other_id))
        other_fact = other_result.scalar_one_or_none()

        output.append({
            "relationship_id": rel.id,
            "relation": rel.relation,
            "explanation": rel.explanation,
            "explanation_basis": rel.explanation_basis,
            "similarity_score": rel.similarity_score,
            "other_fact": _fact_to_dict(other_fact, full=True) if other_fact else None,
        })

    return output


# ── Relationship endpoints ────────────────────────────────────────────────────

@app.get("/relationships", summary="Browse all relationships system-wide")
async def list_relationships(
    relation_type: Optional[str] = Query(
        None,
        alias="type",
        description="Filter by: CORROBORATES, CONTRADICTS, RECONCILABLE, UNRELATED",
    ),
    limit: int = Query(50, le=500),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
):
    query = select(Relationship).order_by(Relationship.created_at.desc())

    if relation_type:
        query = query.where(Relationship.relation == relation_type.upper())

    query = query.limit(limit).offset(offset)
    result = await session.execute(query)
    rels = result.scalars().all()

    output = []
    for rel in rels:
        fact_a_result = await session.execute(select(Fact).where(Fact.id == rel.fact_id_a))
        fact_b_result = await session.execute(select(Fact).where(Fact.id == rel.fact_id_b))
        fact_a = fact_a_result.scalar_one_or_none()
        fact_b = fact_b_result.scalar_one_or_none()

        output.append({
            "relationship_id": rel.id,
            "relation": rel.relation,
            "explanation": rel.explanation,
            "explanation_basis": rel.explanation_basis,
            "similarity_score": rel.similarity_score,
            "fact_a": _fact_to_dict(fact_a) if fact_a else None,
            "fact_b": _fact_to_dict(fact_b) if fact_b else None,
            "created_at": rel.created_at.isoformat() if rel.created_at else None,
        })

    return output


# ── Stats endpoint ────────────────────────────────────────────────────────────

@app.get("/stats", summary="System-wide statistics")
async def get_stats(session: AsyncSession = Depends(get_session)):
    doc_count = await session.scalar(select(func.count(Document.id)))
    fact_count = await session.scalar(select(func.count(Fact.id)))
    rel_count = await session.scalar(select(func.count(Relationship.id)))

    # Relationship breakdown
    rel_breakdown = {}
    for rtype in ["CORROBORATES", "CONTRADICTS", "RECONCILABLE", "UNRELATED"]:
        count = await session.scalar(
            select(func.count(Relationship.id)).where(Relationship.relation == rtype)
        )
        rel_breakdown[rtype.lower()] = count

    # Fact type distribution
    ft_result = await session.execute(
        select(Fact.fact_type, func.count(Fact.id).label("n"))
        .group_by(Fact.fact_type)
        .order_by(text("n DESC"))
    )
    fact_types = {row[0] or "unknown": row[1] for row in ft_result}

    return {
        "documents": doc_count,
        "facts": fact_count,
        "relationships": rel_count,
        "relationship_breakdown": rel_breakdown,
        "fact_type_distribution": fact_types,
        "vector_store_count": vector_store.get_fact_count(),
    }


# ── Helper ────────────────────────────────────────────────────────────────────

def _fact_to_dict(fact: Fact, full: bool = False) -> dict:
    d = {
        "id": fact.id,
        "document_id": fact.document_id,
        "document_name": fact.document_name,
        "page_number": fact.page_number,
        "claim": fact.claim,
        "subject": fact.subject,
        "value": fact.value,
        "time_scope": fact.time_scope,
        "other_scope": fact.other_scope,
        "confidence": fact.confidence,
        "fact_type": fact.fact_type,
        "extraction_notes": fact.extraction_notes,
        "created_at": fact.created_at.isoformat() if fact.created_at else None,
    }
    if full:
        d["quote"] = fact.quote
        d["quote_start"] = fact.quote_start
        d["quote_end"] = fact.quote_end
        d["chunk_id"] = fact.chunk_id
        d["metadata_json"] = fact.metadata_json
    return d
