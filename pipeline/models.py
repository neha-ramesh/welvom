"""Schema definitions. This is the contract for Step 2's output.

Everything downstream (Step 3: clip selection) reads this shape, so it lives
in one place and is validated with pydantic rather than trusted.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


ContentType = Literal[
    "introduction",
    "personal_story",
    "discussion",
    "explanation",
    "advice",
    "question_answer",
    "anecdote",
    "outro",
    "filler",
]


class Metadata(BaseModel):
    video_id: str
    source: str = "local"
    duration: float
    language: Optional[str] = None
    title: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    processed_at: str


class Word(BaseModel):
    word: str
    start: float
    end: float
    score: Optional[float] = None
    # True if repair.py had to correct this word's timing. Step 3 should not
    # use suspect words as snap targets.
    suspect: bool = False


class TranscriptSegment(BaseModel):
    """One chunk as emitted by Whisper. `index` is added by us so the LLM can
    reference chunks by number instead of inventing timestamps."""

    index: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


class Transcript(BaseModel):
    language: str
    segments: list[TranscriptSegment]
    words: list[Word] = Field(default_factory=list)

    @property
    def full_text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments)


class SemanticSegment(BaseModel):
    start: float
    end: float
    start_chunk: int
    end_chunk: int
    topic: str
    summary: str
    content_type: ContentType

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: float, info) -> float:
        start = info.data.get("start")
        if start is not None and v <= start:
            raise ValueError(f"end ({v}) must be greater than start ({start})")
        return v


class LLMSegmentProposal(BaseModel):
    """What we ask the LLM for. Note: NO timestamps. Only chunk indices.

    This is deliberate. LLMs hallucinate numeric timestamps constantly. Chunk
    indices are verifiable against the transcript, so we ask for those and
    derive the real timestamps ourselves.
    """

    start_chunk: int
    end_chunk: int
    topic: str
    summary: str
    content_type: str


class VideoAnalysis(BaseModel):
    metadata: Metadata
    transcript: Transcript
    semantic_segments: list[SemanticSegment]
