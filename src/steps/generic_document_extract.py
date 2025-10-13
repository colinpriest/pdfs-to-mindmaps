# =============================
# FILE: src/steps/generic_document_extract.py
# =============================
from typing import List
from ..llm.client import chat_structured
from ..llm.schemas import GenericExtraction, ChunkEvidence

_GENERIC_SYS = (
    "You analyze document chunks and extract main topics and key concepts/approaches. "
    "This works for any type of document - business reports, manuals, articles, etc. "
    "Extract 3–6 main topics and 1–5 key concepts or approaches mentioned. "
    "Cite evidence by chunk_id. Return JSON conforming to schema."
)

_GENERIC_USER_TPL = (
    "Document id: {pdf_id}. Given chunk list with (chunk_id, section, text), extract:\n"
    "1) topics: id, label, evidence_chunks (list of chunk_id + optional rationale)\n"
    "2) concepts: canonical name, variants, evidence_chunks.\n"
    "Keep it concise; prefer Introduction/Main Content evidence.\n\n"
    "CHUNKS JSONL:\n{jsonl}\n"
)

def extract_generic_document(pdf_id: str, chunks_with_sections: List[dict]) -> GenericExtraction:
    """
    Extracts topics and concepts from any type of document using generic prompts.
    
    Args:
        pdf_id: Unique identifier for the document
        chunks_with_sections: List of chunks with section labels
        
    Returns:
        GenericExtraction with topics and concepts
    """
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
    
    return chat_structured(
        response_model=GenericExtraction,
        system=_GENERIC_SYS,
        user=_GENERIC_USER_TPL.format(pdf_id=pdf_id, jsonl=jsonl),
        temperature=0.1
    )
