from typing import List, Dict
from ..llm.client import chat_structured
from ..llm.schemas import TextRelevance


_RELINK_SYS = (
    "You are a research assistant. Your task is to determine if a given text excerpt is relevant to a specific research topic."
    "The topic is defined by a label and a summary. The excerpt is from a research paper."
    "Respond with a structured JSON object indicating whether the text is relevant and provide a confidence score."
)

_RELINK_USER_TPL = (
    "TOPIC LABEL:\n{topic_label}\n\n"
    "TOPIC SUMMARY:\n{topic_summary}\n\n"
    "TEXT EXCERPT:\n{text_excerpt}\n\n"
    "Based on the provided information, is the text excerpt relevant to the research topic?"
)


def find_relevant_papers_for_topic(
    topic_label: str, topic_summary: str, papers_content: Dict[str, List[str]]
) -> List[str]:
    """
    Finds relevant papers for a topic by checking text excerpts from each paper.

    Args:
        topic_label: The label of the topic.
        topic_summary: A summary of the topic.
        papers_content: A dictionary where keys are paper IDs and values are lists of text chunks.

    Returns:
        A list of paper IDs that are relevant to the topic.
    """
    relevant_paper_ids = []
    for paper_id, chunks in papers_content.items():
        # Check a few chunks from the paper for relevance
        for chunk in chunks[:3]:  # Check first 3 chunks for efficiency
            chunk_text = chunk["text"] if isinstance(chunk, dict) else chunk
            prompt = _RELINK_USER_TPL.format(
                topic_label=topic_label,
                topic_summary=topic_summary,
                text_excerpt=chunk_text,
            )
            try:
                relevance_check = chat_structured(TextRelevance, _RELINK_SYS, prompt)
                if relevance_check.is_relevant and relevance_check.confidence >= 0.75:
                    relevant_paper_ids.append(paper_id)
                    break  # Move to the next paper once relevance is confirmed
            except Exception as e:
                print(f"Could not check relevance for paper {paper_id} and topic {topic_label}: {e}")
                continue
    return relevant_paper_ids
