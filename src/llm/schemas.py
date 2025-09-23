# =============================
# FILE: src/llm/schemas.py
# =============================
from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class TextRelevance(BaseModel):
    """Schema for checking if a text is relevant to a topic."""
    is_relevant: bool = Field(..., description="True if the text is relevant to the topic.")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0) of the relevance assessment.")


class SectionLabel(BaseModel):
    """Schema for labeling a section of a research paper."""
    section: str = Field(..., description="The most likely section label (e.g., 'Abstract', 'Introduction', 'Methods', 'Results', 'Conclusion', 'References').")
    confidence: float = Field(ge=0, le=1)


class TechniqueEnrichment(BaseModel):
    """Schema for enriching a technique with a summary, how it works, and its relevance."""
    how_it_works: List[str] = Field(..., description="5 bullet points explaining how the technique works.")
    relevance: List[str] = Field(..., description="3-5 bullet points explaining the technique's relevance to the topics.")
    summary: str = Field(..., description="A 1-paragraph summary of the technique and its relevance.")


class PaperEnrichment(BaseModel):
    """Schema for enriching a paper with a summary and its top topics."""
    summary: str = Field(..., description="A 1-paragraph summary of the paper, its importance, relevance, and implications.")
    topics: List[str] = Field(..., description="The top 2-3 topics the paper is related to.")


class TopicIdeas(BaseModel):
    """Schema for extracting key ideas and concepts from a topic."""
    ideas: List[str] = Field(..., description="A list of 3-5 key ideas or concepts related to the topic.")


class ChunkEvidence(BaseModel):
    """Schema for chunk-level evidence for a topic or technique."""
    chunk_id: str = Field(..., description="The chunk ID from the paper.")
    rationale: Optional[str] = None

class PaperTopic(BaseModel):
    id: str
    label: str
    evidence_chunks: List[ChunkEvidence]

class PaperTechnique(BaseModel):
    canonical: str
    variants: List[str] = []
    evidence_chunks: List[ChunkEvidence] = []

class PaperExtraction(BaseModel):
    pdf_id: str
    topics: List[PaperTopic]
    techniques: List[PaperTechnique]

class CorpusTopic(BaseModel):
    id: str
    label: str
    member_topic_ids: List[str]

class CorpusTopics(BaseModel):
    topics: List[CorpusTopic]

class TechniqueGroup(BaseModel):
    canonical: str
    variants: List[str]

class NormalizedTechniques(BaseModel):
    groups: List[TechniqueGroup]
