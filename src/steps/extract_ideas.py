from typing import List
from ..llm.client import chat_structured
from ..llm.schemas import TopicIdeas

_IDEAS_SYS = (
    "You are a research analyst. Your task is to distill the key ideas or concepts from a research topic summary."
)

_IDEAS_USER_TPL = (
    "TOPIC:\n{topic_label}\n\n"
    "SUMMARY:\n{topic_summary}\n\n"
    "Based on the topic and its summary, please extract the 3-5 most important ideas or concepts. "
    "Each idea should be a short, concise phrase."
)

def extract_ideas_for_topic(topic_label: str, topic_summary: str) -> TopicIdeas:
    """
    Extracts key ideas and concepts for a topic using an LLM.
    """
    prompt = _IDEAS_USER_TPL.format(topic_label=topic_label, topic_summary=topic_summary)
    
    return chat_structured(
        response_model=TopicIdeas,
        system=_IDEAS_SYS,
        user=prompt,
        temperature=0.1
    )




