# =============================
# FILE: src/llm/schemas.py
# =============================
from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class TextRelevance(BaseModel):
    """Schema for checking if a text is relevant to a topic."""
    is_relevant: bool = Field(..., description="True if the text is relevant to the topic.")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0) of the relevance assessment.")


class MergeDecision(BaseModel):
    """Schema for deciding whether to merge a subtopic into a main topic."""
    should_merge: bool = Field(..., description="True if the subtopic should be merged into the main topic.")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0) of the merge decision.")
    suggested_label: Optional[str] = Field(
        None,
        description="Optional improved label for the merged topic if a merge is suggested.",
    )


class ContentType(BaseModel):
    """Schema for classifying document content type."""
    content_type: Literal["scientific_paper", "other"] = Field(..., description="The type of document content.")
    confidence: float = Field(ge=0, le=1, description="Confidence score (0.0 to 1.0) of the classification.")
    reasoning: str = Field(..., description="Brief explanation of why this classification was chosen.")


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


# Generic schemas for non-scientific documents
class GenericSectionLabel(BaseModel):
    """Schema for labeling sections in any type of document."""
    section: str = Field(..., description="The most likely section label (e.g., 'Introduction', 'Main Content', 'Summary', 'Conclusion', 'Other').")
    confidence: float = Field(ge=0, le=1)


class GenericTopic(BaseModel):
    """Schema for topics extracted from any type of document."""
    id: str
    label: str
    evidence_chunks: List[ChunkEvidence]


class GenericConcept(BaseModel):
    """Schema for concepts/approaches extracted from any type of document."""
    canonical: str
    variants: List[str] = []
    evidence_chunks: List[ChunkEvidence] = []


class GenericExtraction(BaseModel):
    """Schema for extracting topics and concepts from any type of document."""
    pdf_id: str
    topics: List[GenericTopic]
    concepts: List[GenericConcept]
