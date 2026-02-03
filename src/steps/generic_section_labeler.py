# =============================
# FILE: src/steps/generic_section_labeler.py
# =============================
from ..llm.client import chat_structured
from ..llm.schemas import GenericSectionLabel

_GENERIC_SECTION_SYS = (
    "You label document text with a section type that works for any kind of document. "
    "Return a JSON object strictly matching the schema."
)

_GENERIC_SECTION_USER_TPL = (
    "Label this text as one of: Introduction, Main Content, Summary, Conclusion, Other.\n"
    "Return JSON with fields: {{section, confidence}}.\n\n"
    "Text:\n" + "```\n{chunk}\n```"
)

def label_generic_section(chunk_text: str) -> GenericSectionLabel:
    """
    Labels a chunk of text with a generic section type suitable for any document.
    
    Args:
        chunk_text: The text content to label
        
    Returns:
        GenericSectionLabel with section type and confidence
    """
    return chat_structured(
        response_model=GenericSectionLabel,
        system=_GENERIC_SECTION_SYS,
        user=_GENERIC_SECTION_USER_TPL.format(chunk=chunk_text),
        temperature=0.1
    )

