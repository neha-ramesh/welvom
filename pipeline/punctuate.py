"""Insert missing sentence punctuation into a transcript.

Why this exists: Whisper (via Groq) frequently returns the opening stretch of a
transcript with no punctuation at all. In the test file, chunks 0-13 — the first
100 seconds — contain not one full stop. Everything after chunk 14 is punctuated
normally.

That matters because clip cutting snaps to sentence boundaries, and sentence
boundaries are detected from punctuation. No punctuation means no boundaries,
which means clips in that region open mid-sentence. Two of five clips in the
best run so far were flagged "no sentence edge found", both in that window.

The fix is a cheap LLM pass that adds punctuation WITHOUT changing any words.
Word timings stay attached because we re-align by position: token N of the
repaired text is token N of the original, just possibly with a period on it.

Costs a fraction of a cent per video on a small model.
"""

from __future__ import annotations

import re

from .llm import complete_json
from .models import Transcript, TranscriptSegment

# A chunk is considered unpunctuated if it has no sentence-ending mark.
_END_MARKS = (".", "?", "!")

SYSTEM_PROMPT = """You add punctuation to raw speech-to-text output.

You receive numbered transcript chunks with no punctuation. Return the same
chunks with sentence punctuation added.

Absolute rules:
- Do NOT add, remove, reorder, or correct any word. Not one.
- Do NOT fix grammar, spelling, or transcription errors. If it says "before you
  be sleep", leave it as "before you be sleep".
- Only add: full stops, question marks, commas, and capitalisation of the first
  letter after a sentence break.
- A chunk may contain several sentences, or be the middle of one. If a chunk does
  not end a sentence, do not force punctuation onto its end.

The word count of every chunk must be identical to what you were given.

Return ONLY JSON: {"chunks": [{"index": int, "text": str}]}"""


SCHEMA = {
    "type": "object",
    "properties": {
        "chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["index", "text"],
            },
        }
    },
    "required": ["chunks"],
}


def _words_only(text: str) -> list[str]:
    """Bare word tokens, punctuation and case stripped — for comparison."""
    return re.sub(r"[^a-z0-9\s']", " ", text.lower()).split()


def _has_sentence_end(text: str) -> bool:
    return any(m in text for m in _END_MARKS)


def find_unpunctuated(transcript: Transcript) -> list[int]:
    """Chunk indices with no sentence-ending punctuation."""
    return [s.index for s in transcript.segments if not _has_sentence_end(s.text)]


def repunctuate(transcript: Transcript, verbose: bool = True) -> tuple[Transcript, int]:
    """Add punctuation to unpunctuated chunks. Returns (transcript, n_fixed).

    Only touches chunks that need it. If the whole transcript is already
    punctuated this makes no API call at all.
    """
    targets = find_unpunctuated(transcript)
    if not targets:
        if verbose:
            print("      punctuation: nothing to repair")
        return transcript, 0

    block = "\n".join(
        f"[{i}] {transcript.segments[i].text}" for i in targets
    )
    try:
        data = complete_json(
            SYSTEM_PROMPT,
            block,
            schema=SCHEMA,
            list_key="chunks",
            required_field="index",
            temperature=0.0,
        )
    except Exception as e:
        if verbose:
            print(f"      punctuation: repair failed, continuing ({e})")
        return transcript, 0

    fixed = 0
    by_index = {s.index: s for s in transcript.segments}

    for item in data.get("chunks", []):
        try:
            idx = int(item["index"])
            new_text = str(item["text"]).strip()
        except (KeyError, TypeError, ValueError):
            continue
        seg = by_index.get(idx)
        if seg is None or not new_text:
            continue

        # Reject anything that changed the words. The model is only allowed to
        # add punctuation; if word count or content differs, its output is
        # unusable because the word timings would no longer line up.
        if _words_only(new_text) != _words_only(seg.text):
            if verbose:
                print(f"      punctuation: chunk {idx} altered words, skipped")
            continue

        seg.text = new_text
        fixed += 1

    if verbose:
        print(f"      punctuation: repaired {fixed}/{len(targets)} chunks")
    return transcript, fixed


def apply_to_words(transcript: Transcript) -> int:
    """Copy punctuation from repaired chunk text onto the word list.

    Sentence detection reads `transcript.words`, not the chunk text, so the
    punctuation has to be pushed down to individual words or the repair has no
    effect on cutting.

    Alignment is positional: the Nth word token of a chunk maps to the Nth word
    entry inside that chunk's time range. That holds because the repair pass
    rejects any output whose word sequence changed.
    """
    updated = 0
    for seg in transcript.segments:
        tokens = seg.text.split()
        in_range = [
            w for w in transcript.words if seg.start - 0.01 <= w.start < seg.end + 0.01
        ]
        if len(tokens) != len(in_range):
            continue  # counts disagree; leave this chunk's words alone
        for tok, w in zip(tokens, in_range):
            stripped = tok.rstrip()
            if stripped.endswith(_END_MARKS) and not w.word.rstrip().endswith(_END_MARKS):
                w.word = w.word.rstrip() + stripped[-1]
                updated += 1
    return updated
