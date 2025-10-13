# =============================
# FILE: src/steps/content_type_detector.py
# =============================
from typing import List
from ..llm.client import chat_structured
from ..llm.schemas import ContentType

_CONTENT_TYPE_SYS = (
    "You are a document classifier. Your task is to determine whether a document is a scientific academic paper or something else. "
    "Consider factors like: presence of abstract, methodology sections, citations, research questions, experimental design, "
    "academic writing style, and formal structure typical of academic publications."
)

_CONTENT_TYPE_USER_TPL = (
    "Analyze the following document content and classify it as either 'scientific_paper' or 'other'. "
    "Provide your reasoning for the classification.\n\n"
    "DOCUMENT CONTENT (first ~2000 words):\n{content}\n\n"
    "Classification criteria:\n"
    "- scientific_paper: Contains abstract, methodology, results, citations, research questions, academic writing style\n"
    "- other: Business documents, reports, manuals, fiction, news articles, or any non-academic content"
)

def detect_content_type(content: str) -> ContentType:
    """
    Detects whether a document is a scientific academic paper or another type of document.
    
    Args:
        content: The text content of the document (first ~2000 words)
        
    Returns:
        ContentType object with classification, confidence, and reasoning
    """
    # Truncate content to avoid token limits
    truncated_content = content[:2000] if len(content) > 2000 else content
    
    return chat_structured(
        response_model=ContentType,
        system=_CONTENT_TYPE_SYS,
        user=_CONTENT_TYPE_USER_TPL.format(content=truncated_content),
        temperature=0.1
    )
