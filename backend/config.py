"""
FactLens — centralized configuration via pydantic-settings.
All values come from environment variables / .env file; nothing is hard-coded.
"""
from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM provider ──────────────────────────────────────────────────────────
    llm_provider: str = "gemini"          # "gemini" | "openai"
    gemini_api_key: str = ""
    openai_api_key: str = ""

    # ── Model names ───────────────────────────────────────────────────────────
    extraction_model: str = "gemini-1.5-flash"
    judge_model: str = "gemini-1.5-pro"
    embedding_model: str = "text-embedding-004"

    openai_extraction_model: str = "gpt-4o-mini"
    openai_judge_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Pipeline tuning ───────────────────────────────────────────────────────
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 100
    top_k_candidates: int = 10
    min_similarity: float = 0.70
    max_concurrent_extractions: int = 5

    # ── Storage ───────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./data/factlens.db"
    chroma_persist_dir: str = "./data/chroma"
    pdf_upload_dir: str = "./pdfs"

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def active_extraction_model(self) -> str:
        return self.openai_extraction_model if self.llm_provider == "openai" else self.extraction_model

    @property
    def active_judge_model(self) -> str:
        return self.openai_judge_model if self.llm_provider == "openai" else self.judge_model

    @property
    def active_embedding_model(self) -> str:
        return self.openai_embedding_model if self.llm_provider == "openai" else self.embedding_model


settings = Settings()

# Ensure directories exist at import time
Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
Path(settings.pdf_upload_dir).mkdir(parents=True, exist_ok=True)
# Create data dir for SQLite
Path("./data").mkdir(parents=True, exist_ok=True)
