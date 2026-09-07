"""
PDF ingestion: PDF → page text → overlapping chunks.

Every chunk carries provenance metadata so facts extracted from it
can be traced back to the exact page and character offset.

No document-specific logic lives here — all PDFs go through the same path.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pdfplumber

from .config import settings


@dataclass
class PageText:
    page_number: int          # 1-indexed, matches printed page numbers
    pdf_page_index: int       # 0-indexed internal PDF page index
    text: str
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    page_number: int           # page where chunk *starts*
    page_numbers: list[int]    # all pages spanned by this chunk
    text: str
    start_char: int            # char offset in concatenated doc text
    end_char: int
    token_estimate: int = field(init=False)

    def __post_init__(self) -> None:
        # Rough 1 token ≈ 4 chars heuristic (good enough for chunking)
        self.token_estimate = len(self.text) // 4


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def extract_pages(pdf_path: str | Path) -> list[PageText]:
    """
    Extract per-page text from a PDF using pdfplumber.
    Falls back gracefully if a page yields no text (e.g. scanned image pages).
    
    Returns pages in order; page_number is 1-indexed to match PDF printed numbers.
    Note: Some PDFs (like excerpted reports) may have non-sequential printed page
    numbers. We use the PDF's internal page order (1-indexed) for consistency and
    store the extracted text so UI can display the content alongside the number.
    """
    pages: list[PageText] = []
    pdf_path = Path(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for idx, page in enumerate(pdf.pages):
            raw = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            # Collapse excessive whitespace but preserve paragraph breaks
            cleaned = re.sub(r" {3,}", "  ", raw)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
            pages.append(
                PageText(
                    page_number=idx + 1,
                    pdf_page_index=idx,
                    text=cleaned,
                )
            )

    return pages


def chunk_pages(
    pages: list[PageText],
    document_id: str,
    chunk_size_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[Chunk]:
    """
    Sliding-window chunker over concatenated page text.

    Strategy:
    1. Concatenate all page text with page-break markers that preserve page numbers.
    2. Split into overlapping windows of ~chunk_size_tokens tokens.
    3. Each chunk records which pages it spans so facts can cite correct pages.

    The overlap ensures claims that straddle page boundaries are not lost.
    """
    chunk_size = chunk_size_tokens or settings.chunk_size_tokens
    overlap = overlap_tokens or settings.chunk_overlap_tokens

    chunk_size_chars = chunk_size * 4    # token ≈ 4 chars
    overlap_chars = overlap * 4

    # Build a flat text with page-number annotations (for span tracking)
    segments: list[tuple[int, str]] = []   # (page_number, text)
    for page in pages:
        if page.text.strip():
            segments.append((page.page_number, page.text))

    # Build character position → page_number map
    flat_text = ""
    char_to_page: list[int] = []    # char_to_page[i] = page_number at position i

    for page_num, text in segments:
        start = len(flat_text)
        flat_text += text + "\n\n"
        char_to_page.extend([page_num] * (len(flat_text) - start))

    if not flat_text.strip():
        return []

    chunks: list[Chunk] = []
    pos = 0
    total = len(flat_text)

    while pos < total:
        end = min(pos + chunk_size_chars, total)
        # Try to break at sentence/paragraph boundary
        if end < total:
            # Look for nearest newline or period+space within last 20% of chunk
            search_from = pos + int(chunk_size_chars * 0.8)
            boundary = flat_text.rfind("\n", search_from, end)
            if boundary == -1:
                boundary = flat_text.rfind(". ", search_from, end)
            if boundary != -1:
                end = boundary + 1

        chunk_text = flat_text[pos:end].strip()
        if chunk_text:
            # Determine all page numbers spanned by this chunk
            spanned_pages = sorted(set(char_to_page[pos:end]))
            start_page = char_to_page[pos] if pos < len(char_to_page) else pages[-1].page_number

            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    page_number=start_page,
                    page_numbers=spanned_pages,
                    text=chunk_text,
                    start_char=pos,
                    end_char=end,
                )
            )

        # Advance by chunk_size minus overlap (sliding window)
        advance = max(1, chunk_size_chars - overlap_chars)
        pos += advance

    return chunks


def ingest_pdf(pdf_path: str | Path, document_id: str) -> tuple[list[PageText], list[Chunk]]:
    """
    Full ingestion pipeline: PDF file → (pages, chunks).
    Returns both so callers can inspect page count and chunk count.
    """
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, document_id)
    return pages, chunks
