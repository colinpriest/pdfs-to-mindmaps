# =============================
# FILE: src/steps/normalize_techniques.py
# =============================
from typing import List
from ..llm.client import chat_structured
from ..llm.schemas import NormalizedTechniques

_TECH_SYS = (
    "You normalize technique names into canonical groups with variants. JSON only per schema."
)

_TECH_USER_TPL = (
    "Normalize these technique candidates into canonical groups with variants. "
    "Keep only methods/algorithms/models/tests/estimators.\n"
    "INPUT (one per line):\n{lines}\n"
)

def normalize_techniques(candidates: List[str]) -> NormalizedTechniques:
    return chat_structured(NormalizedTechniques, _TECH_SYS, _TECH_USER_TPL.format(lines="\n".join(candidates)))