"""Step 2c — semantic segmentation.

The one design decision that matters here: we do NOT ask the LLM for timestamps.

Ask a model for "start: 421.3" and it will confidently produce a number that
looks right and isn't. Instead we hand it numbered transcript chunks and ask
which chunks group together. Chunk indices are verifiable — we check them
against the transcript and derive the real timestamps from the chunks the model
picked. Timestamp hallucination becomes structurally impossible rather than
something you hope the prompt prevents.
"""

from __future__ import annotations

from .llm import complete_json
from .models import LLMSegmentProposal, SemanticSegment, Transcript

VALID_CONTENT_TYPES = {
    "introduction",
    "personal_story",
    "discussion",
    "explanation",
    "advice",
    "question_answer",
    "anecdote",
    "outro",
    "filler",
}

SYSTEM_PROMPT = """You segment video transcripts into coherent topical sections.

You receive numbered transcript chunks. Group CONSECUTIVE chunks that develop a \
single idea into one semantic segment.

Rules:
- Every chunk index from 0 to N-1 must belong to exactly one segment. No gaps, \
no overlaps.
- Segments must be contiguous: each segment's start_chunk is the previous \
segment's end_chunk + 1.
- A segment is a complete thought, not a fixed length. A tangent that resolves \
in two chunks is its own segment; a story that runs for thirty chunks is one \
segment.
- Break where the subject genuinely shifts, not on a timer.
- topic: 2-4 words, lowercase, snake_case.
- summary: one sentence, third person, describing what is actually said.
- content_type: exactly one of introduction, personal_story, discussion, \
explanation, advice, question_answer, anecdote, outro, filler.
- Use "filler" for housekeeping, sponsor reads, dead air, and small talk. Be \
willing to use it — marking weak material as filler is more useful than \
flattering it.

Return ONLY a JSON object: {"segments": [{"start_chunk": int, "end_chunk": int, \
"topic": str, "summary": str, "content_type": str}]}
No markdown fences, no commentary. Never use the double-quote character inside a
string value — use single quotes if you need to quote speech."""


def _format_chunks(transcript: Transcript) -> str:
    """Compact representation for the LLM.

    Word-level timestamps are deliberately excluded — they'd multiply the token
    count for no benefit, since the model reasons about meaning, not milliseconds.
    """
    lines = []
    for seg in transcript.segments:
        lines.append(f"[{seg.index}] ({seg.start:.1f}s) {seg.text}")
    return "\n".join(lines)


def _call_llm(chunk_block: str, total_chunks: int) -> list[LLMSegmentProposal]:
    data = complete_json(
        SYSTEM_PROMPT,
        f"Transcript has {total_chunks} chunks, indices 0 to "
        f"{total_chunks - 1}.\n\n{chunk_block}",
        list_key="segments",
        required_field="start_chunk",
        temperature=0.2,
    )
    return [LLMSegmentProposal(**s) for s in data.get("segments", [])]


def _repair(
    proposals: list[LLMSegmentProposal], total: int
) -> list[LLMSegmentProposal]:
    """Enforce full contiguous coverage.

    Even with clear instructions, models drop chunks and overlap boundaries. This
    is not a sign of a bad prompt, it's just what happens at scale — so we fix it
    deterministically instead of retrying and hoping.
    """
    clean = [
        p
        for p in proposals
        if 0 <= p.start_chunk <= p.end_chunk < total
    ]
    clean.sort(key=lambda p: p.start_chunk)

    repaired: list[LLMSegmentProposal] = []
    cursor = 0

    for p in clean:
        if p.end_chunk < cursor:
            continue  # fully swallowed by a previous segment
        start = max(p.start_chunk, cursor)
        if start > p.end_chunk:
            continue
        if start > cursor:
            # Gap: attach the orphaned chunks to this segment rather than
            # inventing a segment we have no summary for.
            start = cursor
        content_type = (
            p.content_type if p.content_type in VALID_CONTENT_TYPES else "discussion"
        )
        repaired.append(
            LLMSegmentProposal(
                start_chunk=start,
                end_chunk=p.end_chunk,
                topic=p.topic.strip().lower().replace(" ", "_")[:60] or "untitled",
                summary=p.summary.strip(),
                content_type=content_type,
            )
        )
        cursor = p.end_chunk + 1

    if cursor < total and repaired:
        repaired[-1].end_chunk = total - 1
    elif cursor < total:
        repaired.append(
            LLMSegmentProposal(
                start_chunk=0,
                end_chunk=total - 1,
                topic="unsegmented",
                summary="Segmentation failed; whole transcript as one segment.",
                content_type="discussion",
            )
        )

    return repaired


def segment_transcript(transcript: Transcript) -> list[SemanticSegment]:
    total = len(transcript.segments)
    if total == 0:
        return []

    proposals = _call_llm(_format_chunks(transcript), total)
    proposals = _repair(proposals, total)

    # Derive real timestamps from the chunks the model actually chose.
    result: list[SemanticSegment] = []
    for p in proposals:
        first = transcript.segments[p.start_chunk]
        last = transcript.segments[p.end_chunk]
        result.append(
            SemanticSegment(
                start=first.start,
                end=last.end,
                start_chunk=p.start_chunk,
                end_chunk=p.end_chunk,
                topic=p.topic,
                summary=p.summary,
                content_type=p.content_type,
            )
        )
    return result