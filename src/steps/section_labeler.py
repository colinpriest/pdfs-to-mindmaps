# =============================
# FILE: src/steps/section_labeler.py
# =============================
from typing import Dict
from ..llm.client import chat_structured
from ..llm.schemas import SectionLabel

_SECTION_SYS = (
    "You label scientific text with a section type. "
    "Return a JSON object strictly matching the schema."
)

_SECTION_USER_TPL = (
    "Label this text as one of: Abstract, Introduction, Methods, Results, Discussion, Other.\n"
    "Return JSON with fields: {{section, confidence}}.\n\n"   # <-- braces escaped
    "Text:\n" + "```\n{chunk}\n```"
)


def label_section(chunk_text: str) -> SectionLabel:
    return chat_structured(SectionLabel, _SECTION_SYS, _SECTION_USER_TPL.format(chunk=chunk_text))
