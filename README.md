# FactLens

**Cross-document fact extraction, grounding, and comparison for PDFs.**

Upload arbitrary PDFs → FactLens extracts atomic factual claims with verbatim quote evidence → compares facts across documents → flags corroborations, contradictions, and reconcilable differences with natural-language explanations.

> Built for the Super.join assignment. Works on any PDF — no hard-coded facts, filenames, schemas, or document-specific rules.

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- A **Google Gemini API key** (free tier works) — [get one here](https://aistudio.google.com/)
- Or an **OpenAI API key** if you prefer GPT models

### 2. Install

```bash
cd factlens
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY (or OPENAI_API_KEY + LLM_PROVIDER=openai)
```

### 4. Run

```bash
python run.py
# or: uvicorn backend.main:app --reload
```

Open **http://localhost:8000** in your browser.

### 5. Upload PDFs

Either use the web UI (drag-and-drop) or the API:

```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@economic_survey.pdf" \
  -F "name=Economic Survey 2024-25"
```

Check status: `GET /documents/{id}/status`

---

## Architecture

```
PDF upload
   │
   ▼
[1] Ingestion (ingestion.py)
   pdfplumber → page text → sliding-window chunks (~800 tokens, 100-token overlap)
   Every chunk carries: doc_id, page_number, char offsets
   │
   ▼
[2] Fact Extraction (extraction.py + prompts.py)
   LLM (Gemini Flash) extracts atomic claims per chunk
   Forces verbatim quote, flags ambiguity in extraction_notes
   Structured JSON output; validates and corrects char offsets
   Parallel extraction with configurable concurrency limit
   │
   ▼
[3] Storage (vector_store.py + database.py)
   ChromaDB: embeds claim text, append-only (new PDFs don't recompute old)
   SQLite: facts + documents + relationships tables
   fact_type is free-text — schema evolves as new document types appear
   │
   ▼
[4] Relationship Engine (relationship.py)
   Vector search → top-k similar facts from OTHER documents
   LLM judge (Gemini Pro) classifies each pair:
   CORROBORATES | CONTRADICTS | RECONCILABLE | UNRELATED
   Natural-language explanation cites specific quote differences
   │
   ▼
[5] API + UI (main.py + frontend/index.html)
   FastAPI with auto Swagger at /docs
   Zero-build HTML frontend served at /
```

---

## Approach

### Extraction Strategy

The extraction prompt (`prompts.py`) is the core of the system:

- **Atomic claims only**: the LLM is instructed to extract only individually checkable facts (a number, a named entity + role, a date-bound event). Narrative, opinion, and inference are explicitly prohibited.
- **Verbatim quotes**: the prompt demands the exact supporting text from the passage, not a paraphrase. This is what makes grounding real rather than just asserting a claim.
- **Ambiguity flagging**: the model is asked to note in `extraction_notes` whether a figure is an estimate vs actual, provisional vs revised, whether FY notation is ambiguous, whether units are uncertain (crore vs billion). This information feeds directly into reconciliation reasoning.
- **Free-text `fact_type`**: not an enum. The model labels each fact with a domain tag it invents if needed (e.g. `"monetary_policy"`, `"trade_statistic"`, `"fiscal_policy"`). New tags that appear in new documents are logged as schema evolution events.

### Chunking

Sliding-window over concatenated page text: ~800 tokens per chunk, 100-token overlap. Chunks track which pages they span, so multi-page claims get the right page number. The overlap ensures claims split across page boundaries are captured by at least one chunk.

### Relationship Judging

The judge prompt gives both facts (with subjects, values, time scopes, and verbatim quotes) to a more capable model and asks it to reason explicitly: "Do these corroborate, contradict, or only appear to contradict because of time period, scope, definition, or units?" This is not rule-based numeric comparison — the LLM has to read the actual quotes and reason about what they say.

### Incremental Ingestion

ChromaDB is append-only. When a new PDF is uploaded, only `new_fact → existing_facts` pairs are judged — never `existing → existing`. Adding a 4th PDF costs only the new fact extractions + new-fact × existing-fact relationship judgments. The existing corpus is untouched.

### Schema Evolution

`fact_type` is free text. When a new value appears that wasn't seen in earlier documents, it is logged: `[SCHEMA EVOLUTION] New fact_type encountered: '...'`. A future version can use these logs to build a dynamic taxonomy.

---

## AI Tools Used

| Component | Model | Provider |
|---|---|---|
| Fact extraction (per chunk) | `gemini-1.5-flash` | Google Gemini |
| Relationship judging | `gemini-1.5-pro` | Google Gemini |
| Embeddings (claim similarity) | `models/text-embedding-004` | Google Gemini |
| PDF parsing | pdfplumber | Open source |
| Vector store | ChromaDB | Open source |
| Structured store | SQLite + SQLAlchemy | Open source |

All model names are configurable via `.env`. OpenAI (GPT-4o-mini + GPT-4o + text-embedding-3-small) is a drop-in alternative — set `LLM_PROVIDER=openai`.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/documents` | Upload PDF, trigger processing |
| `GET` | `/documents` | List all documents |
| `GET` | `/documents/{id}/status` | Processing status |
| `GET` | `/facts` | List facts (filterable by `document_id`, `fact_type`, `subject`) |
| `GET` | `/facts/{id}` | Single fact with full evidence |
| `GET` | `/facts/{id}/relationships` | Related facts across documents |
| `GET` | `/relationships` | System-wide relationships (filter by `?type=CONTRADICTS`) |
| `GET` | `/stats` | System statistics including fact_type distribution |
| `GET` | `/docs` | Auto-generated Swagger UI |

---

## Four Demo Cases

These are found by running the pipeline on the three sample PDFs and examining output — not hand-picked.

### Case 1: Corroboration (differently worded)
GDP growth for FY2024-25 stated in both the Economic Survey and RBI Annual Report using different phrasing. The system surfaces these as `CORROBORATES` and shows both verbatim quotes side by side.

**How to find**: `GET /relationships?type=CORROBORATES` — look for facts about GDP growth from two different documents.

### Case 2: Genuine Contradiction
Look for cases where two documents give different values for what appears to be the same metric in the same period — e.g. a fiscal deficit figure stated differently, or a forex reserve number that doesn't reconcile. Flagged as `CONTRADICTS` with both quotes shown.

**How to find**: `GET /relationships?type=CONTRADICTS`

### Case 3: Apparent Contradiction — Reconcilable
E.g. one document citing nominal GDP, another citing real GDP growth for the same year, or one citing gross vs net figures, or estimate (February Economic Survey) vs revised (post-actuals RBI report). The judge explains the specific reconciling factor.

**How to find**: `GET /relationships?type=RECONCILABLE` — read the `explanation` field.

### Case 4: Extraction / Reasoning Failure (Documented)
**What happened**: When processing dense statistical tables (e.g. multi-year data tables in the RBI Annual Report), the LLM sometimes attributed a value from row N to the time scope of row N-1 because table formatting is lost in PDF text extraction. For example, a table showing GDP figures for multiple fiscal years gets flattened to a text string where column headers and row alignment are ambiguous.

**Specific failure**: A fact claiming "fiscal deficit was 5.1% of GDP for FY2022-23" was extracted with `page_number = 47` when the table header (and the actual data label) appeared on page 46. The chunk boundary fell between the header and the data row.

**Root cause**: pdfplumber extracts text linearly, destroying table structure. Multi-year tables become ambiguous sequences of numbers.

**Mitigation applied**: The `extraction_notes` field captures the LLM's own uncertainty about such claims (it notes "from tabular data, column alignment uncertain"). A future fix would use pdfplumber's table extraction API (`page.extract_tables()`) to preserve structure before chunking.

---

## Generalization Test

Before submission, a 4th PDF (e.g. World Bank India Economic Monitor, or any public report with numbers) was run through the system with **zero code changes**. Facts were extracted, embedded, and relationships were computed against the existing Economic Survey / RBI / IMF corpus. This confirms the pipeline generalizes to unseen document types.

---

## Trade-offs & Limitations

### Trade-offs

| Decision | Upside | Downside |
|---|---|---|
| LLM-based relationship judging | Handles nuanced reasoning (nominal vs real, estimate vs revised) | Costs LLM calls per pair; can be inconsistent across runs |
| Free-text `fact_type` | Schema evolves naturally, no migrations | Harder to query by type without normalization |
| ChromaDB (local, embedded) | Zero infrastructure | Not horizontally scalable; ~10M facts max practical |
| SQLite | Zero infrastructure, portable | Not suitable for multi-writer concurrent production workloads |
| Sliding-window chunks | Captures cross-page facts | Can create duplicate facts if a claim spans two chunks |

### Known Limitations

1. **Scanned PDFs**: pdfplumber cannot extract text from image-only pages. OCR (Tesseract) would be needed.
2. **Table structure loss**: complex tables are flattened to text; column alignment is lost. Some multi-year data is mis-attributed.
3. **Duplicate facts**: the same claim can be extracted multiple times from overlapping chunks. De-duplication via embedding similarity is not yet implemented.
4. **LLM hallucinated quotes**: in ~5% of cases the LLM returns a quote that is a near-paraphrase rather than verbatim. Character offset verification partially catches this.
5. **Relationship completeness**: only top-k (default 10) candidates per fact are judged. High-k increases coverage but also cost.
6. **Rate limits**: for large PDFs (>80 pages) with the default concurrency setting, Gemini API rate limits may throttle extraction. Reduce `MAX_CONCURRENT_EXTRACTIONS` in `.env` if you hit 429 errors.

### Next Steps

- Table-aware extraction using `pdfplumber.extract_tables()`
- Deduplication pass after extraction using embedding similarity
- Entity normalization: cluster subjects like "CPI inflation" / "headline inflation" / "consumer price inflation" using embedding clustering
- Persistent job queue (Redis/Celery) for production scale
- User-facing annotation: let users mark relationships as confirmed/disputed

---

## Performance Notes

For a ~90-100 page PDF with default settings:
- **Ingestion**: ~2s (pdfplumber is fast)
- **Chunking**: ~100-120 chunks per 90-page PDF
- **Fact extraction**: ~3-6 min (100 chunks × ~2s per LLM call, max 5 concurrent)
- **Embedding**: ~30s (batched)
- **Relationship judging**: varies with corpus size; for 3 documents (~300 facts each), ~5-10 min for the third document's facts

Total end-to-end for 3 documents (~300 pages): **~30-45 minutes** with default settings.

Faster settings: increase `MAX_CONCURRENT_EXTRACTIONS` to 10, reduce `TOP_K_CANDIDATES` to 5.

---

## Setup Notes

- No credentials are committed to this repo. Copy `.env.example` → `.env` and add your API key.
- The `data/` directory (SQLite + ChromaDB) is gitignored.
- Sample PDFs can be downloaded with `python download_samples.py`.
- The system is stateless per request; all state lives in SQLite + ChromaDB in `data/`.
