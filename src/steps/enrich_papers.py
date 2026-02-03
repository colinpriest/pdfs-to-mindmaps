from typing import List, Union, Dict
from ..llm.client import chat_structured
from ..llm.schemas import PaperEnrichment

_ENRICH_SYS = (
    "You are a research assistant. Your task is to summarize a research paper and identify its main topics."
)

_ENRICH_USER_TPL = (
    "PAPER TEXT (first ~1200 words):\n{paper_text}\n\n"
    "Based on the text, please provide the following:\n"
    "1. A 1-paragraph summary of the paper, covering its importance, relevance, and implications.\n"
    "2. The top 2-3 topics the paper is related to."
)

def enrich_paper(paper_chunks: List[Union[str, Dict[str, str]]]) -> PaperEnrichment:
    """
    Enriches a paper with a summary and topics using an LLM.
    """
    # Combine the first few chunks to get a representative sample of the paper text
    selected_chunks = paper_chunks[:2]
    chunk_texts = [
        chunk["text"] if isinstance(chunk, dict) else chunk for chunk in selected_chunks
    ]
    paper_text = " ".join(chunk_texts)
    
    prompt = _ENRICH_USER_TPL.format(paper_text=paper_text)
    
    return chat_structured(
        response_model=PaperEnrichment,
        system=_ENRICH_SYS,
        user=prompt,
        temperature=0.2
    )





















