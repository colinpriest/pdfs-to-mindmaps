from typing import List
from ..llm.client import chat


_SUMMARY_SYS = (
    "You are a research assistant. "
    "Given a research topic, a list of relevant paper titles, and a list of techniques, "
    "write a concise 1-3 paragraph summary of the topic. "
    "Synthesize the information to explain what the topic is about, drawing connections between the papers and techniques."
)

_SUMMARY_USER_TPL = (
    "Please summarize the following research topic.\n\n"
    "TOPIC:\n{topic_label}\n\n"
    "PAPERS:\n{papers_str}\n\n"
    "TECHNIQUES:\n{techniques_str}"
)


def summarize_topic(topic_label: str, paper_titles: List[str], technique_names: List[str]) -> str:
    """Generates a summary for a topic based on related papers and techniques."""
    papers_str = "- " + "\n- ".join(paper_titles) if paper_titles else "N/A"
    techniques_str = "- " + "\n- ".join(technique_names) if technique_names else "N/A"
    prompt = _SUMMARY_USER_TPL.format(
        topic_label=topic_label,
        papers_str=papers_str,
        techniques_str=techniques_str,
    )
    summary = chat(_SUMMARY_SYS, prompt)
    return summary

