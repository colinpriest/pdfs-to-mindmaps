# =============================
# FILE: src/steps/per_paper_extract.py
# =============================
from typing import List
from ..llm.client import chat_structured
from ..llm.schemas import PaperExtraction, ChunkEvidence

_PAPER_SYS = (
    "You analyze one paper's chunks (with sections). "
    "Extract 3–6 short research topics and 1–5 techniques/methods it uses. "
    "Cite evidence by chunk_id. Return JSON conforming to schema."
)

_PAPER_USER_TPL = (
    "Paper id: {pdf_id}. Given chunk list with (chunk_id, section, text), extract:\n"
    "1) topics: id, label, evidence_chunks (list of chunk_id + optional rationale)\n"
    "2) techniques: canonical, variants, evidence_chunks.\n"
    "Keep it concise; prefer Methods/Abstract evidence.\n\n"
    "CHUNKS JSONL:\n{jsonl}\n"
)

def extract_paper(pdf_id: str, chunks_with_sections: List[dict]) -> PaperExtraction:
    # Prepare a compact JSONL payload for the prompt
    import json
    lines = []
    for ch in chunks_with_sections:
        lines.append(json.dumps({
            "chunk_id": ch["chunk_id"],
            "section": ch["section"],
            "text": ch["text"][:1600]  # trim per chunk to control tokens
        }))
    jsonl = "\n".join(lines)
    return chat_structured(PaperExtraction, _PAPER_SYS, _PAPER_USER_TPL.format(pdf_id=pdf_id, jsonl=jsonl))