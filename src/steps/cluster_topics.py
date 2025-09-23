# =============================
# FILE: src/steps/cluster_topics.py
# =============================
from typing import List
from ..llm.client import chat_structured
from ..llm.schemas import CorpusTopics

_CLUSTER_SYS = (
    "You cluster paper-level topics into corpus topics. "
    "Return 8–20 clusters with clear labels. JSON only per schema."
)

_CLUSTER_USER_TPL = (
    "Cluster these paper-level topics into ~{k} corpus topics. Each item has id and label.\n"
    "For each corpus topic, return: id, label, member_topic_ids (list of input topic ids).\n"
    "INPUT JSONL (id, label):\n{jsonl}\n"
)

def cluster_topics(paper_topic_items: List[dict], k: int = 12) -> CorpusTopics:
    import json
    lines = [json.dumps({"id": it["id"], "label": it["label"]}) for it in paper_topic_items]
    jsonl = "\n".join(lines)
    return chat_structured(CorpusTopics, _CLUSTER_SYS, _CLUSTER_USER_TPL.format(jsonl=jsonl, k=k))
