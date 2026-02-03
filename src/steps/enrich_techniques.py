from typing import List
from ..llm.client import chat_structured
from ..llm.schemas import TechniqueEnrichment

_ENRICH_SYS = (
    "You are a research assistant. Your task is to provide a detailed explanation of a research technique "
    "in the context of the given research topics."
)

_ENRICH_USER_TPL = (
    "TECHNIQUE:\n{technique_name}\n\n"
    "RELEVANT TOPICS:\n- {topics_str}\n\n"
    "Please provide the following information about the technique:\n"
    "1. A 1-paragraph summary of the technique and its relevance to the topics.\n"
    "2. 5 bullet points explaining how the technique works.\n"
    "3. 3-5 bullet points explaining how the technique is relevant to the provided topics."
)

def enrich_technique(technique_name: str, topic_context: List[str]) -> TechniqueEnrichment:
    """
    Enriches a technique with a summary, explanation, and relevance using an LLM.
    """
    topics_str = "\n- ".join(topic_context)
    prompt = _ENRICH_USER_TPL.format(technique_name=technique_name, topics_str=topics_str)
    
    return chat_structured(
        response_model=TechniqueEnrichment,
        system=_ENRICH_SYS,
        user=prompt,
        temperature=0.2
    )






















