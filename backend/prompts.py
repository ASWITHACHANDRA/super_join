"""
FactLens — LLM prompt templates.

Two prompts:
1. EXTRACTION_SYSTEM / build_extraction_prompt — extracts atomic facts from a chunk
2. JUDGE_SYSTEM / build_judge_prompt — judges relationship between two candidate facts

These are the core reasoning components; they must work on ANY document domain.
"""
from __future__ import annotations

EXTRACTION_SYSTEM = """You are a precise fact-extraction engine. Your job is to read a passage from a document and extract every atomic, checkable factual claim it contains.

RULES (follow exactly):
1. Extract only ATOMIC claims: a single number, a named entity + role/attribute, a date-bound event, a measurable state, a policy action. Do NOT extract opinions, predictions without stated evidence, or narrative.
2. Each claim must be independently checkable — if you can't imagine how to verify it from public data, don't extract it.
3. The "quote" field MUST be a verbatim (word-for-word) excerpt from the passage that directly supports the claim. Do NOT paraphrase. The quote should be the shortest passage that unambiguously supports the claim.
4. In "extraction_notes", flag any ambiguity: estimate vs actual, provisional vs final, fiscal-year notation ambiguity, unit uncertainty (crore vs billion), gross vs net, nominal vs real, preliminary vs revised. If none, write null.
5. "fact_type" is a free-text label you assign based on the domain of the claim (e.g. "economic_indicator", "monetary_policy", "fiscal_policy", "demographic", "trade_statistic", "financial_market", "corporate_result", "regulatory_action"). Invent new types as needed — this is NOT an enum.
6. "subject" is the entity or metric being described (e.g. "India real GDP growth", "CPI inflation", "RBI repo rate").
7. "value" is the measured/stated quantity or state (e.g. "6.4%", "Rs 40.92 lakh crore", "unchanged at 6.5%").
8. "time_scope" is the time period the claim refers to (e.g. "FY2024-25", "Q3 FY25", "December 2024"). Use null if no time reference.
9. "confidence" is your confidence (0.0–1.0) that this is a genuine factual claim (not a projection, assumption, or rhetorical statement).
10. Return ONLY a JSON array. No prose before or after.

OUTPUT FORMAT (JSON array, one object per fact):
[
  {
    "claim": "<one-sentence statement of the fact>",
    "subject": "<entity or metric>",
    "value": "<measured quantity or state>",
    "time_scope": "<time period or null>",
    "other_scope": "<sector, region, instrument, etc. or null>",
    "quote": "<verbatim excerpt from the passage>",
    "quote_start": <char offset of quote start within the passage, integer>,
    "quote_end": <char offset of quote end within the passage, integer>,
    "confidence": <float 0.0–1.0>,
    "extraction_notes": "<ambiguity flags or null>",
    "fact_type": "<free-text domain label>"
  }
]

If no atomic facts are present, return an empty array: []
"""


def build_extraction_prompt(chunk_text: str, page_number: int, document_name: str) -> str:
    return f"""Document: {document_name}
Page range starts at: {page_number}

--- PASSAGE BEGIN ---
{chunk_text}
--- PASSAGE END ---

Extract all atomic factual claims from this passage following the rules above. Return ONLY the JSON array."""


JUDGE_SYSTEM = """You are a cross-document fact-comparison engine. You will be given two atomic factual claims extracted from DIFFERENT source documents, along with their verbatim supporting quotes.

Your task: determine the relationship between the two claims and explain your reasoning rigorously.

RELATIONSHIP TYPES:
- CORROBORATES: Both claims assert the same (or materially equivalent) fact for the same subject, time period, scope, and units. Minor wording differences or small revisions (estimate → revised) that don't change the substance count as corroborating.
- CONTRADICTS: The claims assert genuinely different facts for what appears to be the same subject, time period, scope, and units, with no clear reconciling factor from the provided quotes.
- RECONCILABLE: The claims appear to contradict but can be explained by a specific, nameable difference — time period, scope (national vs regional, gross vs net, nominal vs real), units (crore vs billion, percent vs basis points), definition (headline vs core, fiscal year definition), or data vintage (estimate → preliminary → revised → final). You MUST name the specific difference.
- UNRELATED: The claims cover different subjects or are too dissimilar to be meaningfully compared.

RULES:
1. Do NOT assume a contradiction without checking time periods, units, and scope.
2. Do NOT assume corroboration without verifying the time period and scope match.
3. Always cite the specific text from the quotes that drives your reasoning.
4. If RECONCILABLE, list the reconciling factor(s) in explanation_basis.
5. Be honest: if you cannot determine the relationship with reasonable confidence, say UNRELATED and explain why.

OUTPUT FORMAT (JSON object, no prose outside it):
{
  "relation": "CORROBORATES | CONTRADICTS | RECONCILABLE | UNRELATED",
  "explanation": "<2-4 sentence natural language explanation citing specific text from the quotes>",
  "explanation_basis": ["<free-text tag 1>", "<free-text tag 2>", ...]
}

Valid explanation_basis tags (examples — invent new ones as needed):
time_scope_match, time_scope_mismatch, unit_difference, scope_difference,
definition_difference, estimate_vs_revised, gross_vs_net, nominal_vs_real,
value_within_tolerance, value_discrepancy, same_subject_different_period,
fiscal_year_notation_difference
"""


def build_judge_prompt(
    fact_a: dict,
    fact_b: dict,
    doc_a_name: str,
    doc_b_name: str,
) -> str:
    return f"""Compare these two factual claims from different source documents.

=== CLAIM A (from: {doc_a_name}, page {fact_a.get('page_number', '?')}) ===
Claim: {fact_a['claim']}
Subject: {fact_a.get('subject', 'N/A')}
Value: {fact_a.get('value', 'N/A')}
Time scope: {fact_a.get('time_scope', 'N/A')}
Other scope: {fact_a.get('other_scope', 'N/A')}
Quote: "{fact_a.get('quote', 'N/A')}"
Extraction notes: {fact_a.get('extraction_notes', 'none')}

=== CLAIM B (from: {doc_b_name}, page {fact_b.get('page_number', '?')}) ===
Claim: {fact_b['claim']}
Subject: {fact_b.get('subject', 'N/A')}
Value: {fact_b.get('value', 'N/A')}
Time scope: {fact_b.get('time_scope', 'N/A')}
Other scope: {fact_b.get('other_scope', 'N/A')}
Quote: "{fact_b.get('quote', 'N/A')}"
Extraction notes: {fact_b.get('extraction_notes', 'none')}

Determine the relationship between these two claims. Return ONLY the JSON object."""


def build_normalization_prompt(subjects: list[str]) -> str:
    """
    Optional: ask LLM to cluster a list of subject strings into canonical groups.
    Used as a fallback when embedding-based clustering is uncertain.
    """
    subjects_str = "\n".join(f"- {s}" for s in subjects)
    return f"""The following strings are "subject" labels extracted from financial/economic documents.
Group them into clusters where each cluster represents the same underlying metric or entity.
Different phrasings of the same concept should be in the same cluster.

Subjects:
{subjects_str}

Return a JSON array of clusters:
[
  {{
    "canonical": "<canonical name for this cluster>",
    "members": ["<original string 1>", "<original string 2>", ...]
  }}
]

Return ONLY the JSON array."""
