"""
SQLAlchemy async models for FactLens.

Schema is deliberately loose/extensible:
 - facts.metadata_json holds a JSON blob for fields not yet in the fixed columns
 - fact_type and other_scope are free-text, not enums — schema evolves as new
   document types are ingested
 - relationships.explanation_basis is a JSON list of free-text tags
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer,
    String, Text, JSON, Index,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String, nullable=False)          # user-friendly display name
    filename = Column(String, nullable=False)       # original upload filename
    filepath = Column(String, nullable=False)       # path on disk
    status = Column(String, default="pending")      # pending|processing|done|error
    error_message = Column(Text, nullable=True)
    page_count = Column(Integer, nullable=True)
    chunk_count = Column(Integer, nullable=True)
    fact_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)

    facts = relationship("Fact", back_populates="document", cascade="all, delete-orphan")


class Fact(Base):
    """
    Atomic factual claim extracted from a document chunk.

    Fixed columns cover the core structured fields from the spec.
    metadata_json is a catch-all JSON blob for new fields that emerge
    from unseen document types — this is how the schema evolves without
    migrations.
    """
    __tablename__ = "facts"

    id = Column(String, primary_key=True, default=_uuid)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    document_name = Column(String, nullable=False)

    # Evidence / grounding
    page_number = Column(Integer, nullable=True)
    chunk_id = Column(String, nullable=True)       # chunk that produced this fact

    # Core claim fields (all optional at column level — LLM may not always populate)
    claim = Column(Text, nullable=False)            # human-readable atomic claim
    subject = Column(String, nullable=True)         # e.g. "India real GDP growth"
    value = Column(String, nullable=True)           # e.g. "6.4%"
    time_scope = Column(String, nullable=True)      # e.g. "FY2024-25"
    other_scope = Column(Text, nullable=True)       # free text: sector, region, etc.

    # Verbatim evidence
    quote = Column(Text, nullable=True)             # near-verbatim text from source
    quote_start = Column(Integer, nullable=True)    # char offset in page text
    quote_end = Column(Integer, nullable=True)

    # Quality / metadata
    confidence = Column(Float, nullable=True)
    extraction_notes = Column(Text, nullable=True)  # ambiguity flags from LLM
    fact_type = Column(String, nullable=True)       # free-text tag, not an enum

    # Extensible catch-all
    metadata_json = Column(JSON, nullable=True)

    # ChromaDB embedding reference
    embedding_id = Column(String, nullable=True)

    created_at = Column(DateTime, default=_now)

    document = relationship("Document", back_populates="facts")
    relationships_as_a = relationship(
        "Relationship",
        foreign_keys="Relationship.fact_id_a",
        back_populates="fact_a",
        cascade="all, delete-orphan",
    )
    relationships_as_b = relationship(
        "Relationship",
        foreign_keys="Relationship.fact_id_b",
        back_populates="fact_b",
        cascade="all, delete-orphan",
    )


class Relationship(Base):
    """
    Cross-document relationship between two facts.

    relation is one of: CORROBORATES | CONTRADICTS | RECONCILABLE | UNRELATED
    explanation_basis is a JSON list of free-text tags (e.g. ["time_scope_match"])
    """
    __tablename__ = "relationships"

    id = Column(String, primary_key=True, default=_uuid)
    fact_id_a = Column(String, ForeignKey("facts.id"), nullable=False)
    fact_id_b = Column(String, ForeignKey("facts.id"), nullable=False)
    relation = Column(String, nullable=False)           # CORROBORATES | CONTRADICTS | RECONCILABLE | UNRELATED
    explanation = Column(Text, nullable=True)
    explanation_basis = Column(JSON, nullable=True)     # list of free-text tags
    similarity_score = Column(Float, nullable=True)     # cosine sim from vector search

    created_at = Column(DateTime, default=_now)

    fact_a = relationship("Fact", foreign_keys=[fact_id_a], back_populates="relationships_as_a")
    fact_b = relationship("Fact", foreign_keys=[fact_id_b], back_populates="relationships_as_b")


# Indexes for common query patterns
Index("ix_facts_document_id", Fact.document_id)
Index("ix_facts_fact_type", Fact.fact_type)
Index("ix_facts_subject", Fact.subject)
Index("ix_relationships_fact_a", Relationship.fact_id_a)
Index("ix_relationships_fact_b", Relationship.fact_id_b)
Index("ix_relationships_relation", Relationship.relation)
