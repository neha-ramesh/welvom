"""Step 3 — pick the best clips.

Two modes, so you can compare pipelines:

  mode="segments"  Reads the topic groups from Step 2 and looks for clips inside
                   them. Original behaviour.
  mode="direct"    Ignores the topic groups and reads the transcript chunks
                   straight. Groups are used only to reject outros and filler.

Why "direct" exists: on the 6-minute test video the grouping step produced only
four groups, and the first covered 0:00-2:56 — 48% of the runtime. The scorer saw
one huge undifferentiated block and pulled two candidates from it, missing the
strongest moment in the video entirely. Reading chunks directly removes the
dependency on how well the grouping happened to work.

Three design decisions carried over:

1. The model returns chunk INDICES, never timestamps. Ask a model "when does this
   start" and it invents a plausible wrong number. Indices are checkable against
   the transcript, so we look up the real seconds ourselves.

2. Duration is enforced in code. Models cannot count seconds — one returned a
   128-second range when told to produce 45-second clips.

3. Cut points snap to sentence boundaries where we can find them. Chunk edges are
   NOT sentence edges: chunk 33 of the test file begins mid-sentence with "way the
   next day...", so cutting at the chunk edge opened a clip on a fragment.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .llm import complete_json
from .models import Transcript, VideoAnalysis, Word

TARGET_MIN = 28.0   # was 20; anchoring produced 26s clips, too thin for a reel
TARGET_MAX = 55.0   # was 75; direct mode drifted to 61s clips, too long for a reel
IDEAL = 42.0

# How far the start may move forward to land on the hook line.
HOOK_MAX_SHIFT = 20.0
# Small lead-in so the first syllable of the hook is never clipped.
# Lead-in before the first word of a clip, so no syllable is clipped.
#
# A fixed pad is not enough. Whisper's start time for a word after a pause is
# often late — in clip #4 the cut landed on "different," at 267.633 while the
# previous word ended at 266.993, so 0.64s of the actual onset was outside the
# clip and it opened a beat into the word. Clips that cut on a word with a tight
# preceding gap sounded fine; the ones with a big gap did not.
#
# So we take back most of whatever silence precedes the word, within limits.
HOOK_PAD = 0.12         # minimum, when the previous word ends right before
HOOK_PAD_MAX = 0.25     # cap, so we don't drag in the tail of the previous word
GAP_RECLAIM = 0.75      # fraction of the preceding gap to reclaim


def lead_in(t: float, words: list[Word]) -> float:
    """Back the cut point off into whatever silence precedes it."""
    prev_end = max((w.end for w in words if w.end <= t + 0.01), default=None)

    # If the word we're cutting on had its timing repaired, the gap that used to
    # sit before it has been clamped away — repair.py pushes an overlapping word's
    # start up to the previous word's end, so the gap reads as zero even though
    # the speaker's real onset is earlier. Clip #4 cut on a suspect "different,"
    # and opened a beat into the word for exactly this reason. Assume a moderate
    # lead-in rather than trusting a number we know was overwritten.
    at_cut = next((w for w in words if abs(w.start - t) < 0.02), None)
    if at_cut is not None and at_cut.suspect:
        return max(0.0, t - HOOK_PAD_MAX)

    if prev_end is None:
        return max(0.0, t - HOOK_PAD)
    gap = t - prev_end
    if gap <= 0:
        return max(0.0, t - HOOK_PAD)
    pad = min(max(gap * GAP_RECLAIM, HOOK_PAD), HOOK_PAD_MAX)
    return max(0.0, t - pad)

# How far a cut point may move to reach a sentence boundary.
START_FORWARD = 3.5   # trimming a fragment off the front is worth a few seconds
START_BACK = 1.0
END_WINDOW = 2.5

Mode = Literal["segments", "direct"]


# --------------------------------------------------------------------------- #
# Content-type playbooks
# --------------------------------------------------------------------------- #

PLAYBOOKS = {
    "testimonial": """This is a customer testimonial. A clip that works contains:
- the PROBLEM they had before, stated concretely
- their HESITATION or scepticism, if present
- a SPECIFIC result, ideally with a number or a before/after

Reject vague praise. "They were great to work with" is worthless. "We were
losing two days a week to manual invoicing and now it's twenty minutes" is the
clip. Prefer the speaker's own words over any summary.""",
    "founder_talk": """This is a founder or expert talking. A clip that works contains:
- a CLAIM that is contrarian, surprising, or specific enough to disagree with
- the REASONING behind it
- a concrete payoff, example, or number

Reject generic motivation. "Work hard and stay consistent" is worthless. A claim
someone could argue with is the clip.""",
    "interview": """This is an interview. A clip that works is a single exchange
where the answer stands alone without the question needing context. Prefer moments
where the speaker says something they clearly didn't plan to say.""",
    "generic": """A clip that works opens with something that makes a scrolling
viewer stop, delivers one complete idea, and ends on a resolution rather than
trailing off. It must make sense to someone who has not seen the rest of the
video.""",
}


_SHARED_RULES = """
Hard rules:
- A candidate is a CONTIGUOUS range of chunk indices.
- It must be self-contained: understandable with zero surrounding context. If it
  opens with "and so that's why...", it starts too late. If it needs the previous
  question to make sense, widen it or reject it.
- It must end on a resolved thought, not mid-sentence.
- Do not return candidates from housekeeping, intros, sponsor reads, outros, or
  small talk.
- Candidates MAY overlap each other. Overlaps are removed afterwards, so propose
  every good option rather than pre-filtering.

For each candidate:
- start_chunk / end_chunk: integers
- hook: the OPENING line of the clip, copied verbatim from the transcript. This
  must be the very first words spoken in your chunk range, not the most
  interesting line from the middle. If the strongest line is 10 seconds in, then
  START the range there instead and use that line as the hook.
- reason: one sentence on why this stands alone and what makes it land
- has_specifics: true if it contains a number, name, or concrete detail
- self_contained: true if it needs no outside context at all
- score: 0-100. Reserve above 80 for clips you would stake money on.

Return ONLY JSON: {{"candidates": [...]}}
No markdown fences, no commentary.

IMPORTANT: never use the double-quote character inside any string value. If the
transcript contains quoted speech, use single quotes instead. Unescaped double
quotes break the response."""


SEGMENTS_PROMPT = """You find short-form clip candidates inside a long video transcript.

You receive numbered transcript chunks and a list of topical segments.

{playbook}
""" + _SHARED_RULES + """
- Return 8 to 15 candidates so there is something to rank."""


DIRECT_PROMPT = """You find short-form clip candidates inside a long video transcript.

You receive numbered transcript chunks with their timestamps. Work through the
WHOLE transcript from the first chunk to the last. Do not skip the opening — the
strongest moment is often in the first minute.

{playbook}
""" + _SHARED_RULES + """
- Return 10 to 20 candidates.
- Cover the whole runtime. If the transcript spans six minutes you should be
  proposing candidates from the first minute, the middle and the end, not
  clustering them in one stretch.
- Each candidate should span roughly enough chunks to cover 25-50 seconds. Use the
  timestamps shown to judge this. Shorter is safer than longer."""


# Sent to Claude as output_config.format so the API validates the shape itself.
CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_chunk": {"type": "integer"},
                    "end_chunk": {"type": "integer"},
                    "hook": {"type": "string"},
                    "reason": {"type": "string"},
                    "has_specifics": {"type": "boolean"},
                    "self_contained": {"type": "boolean"},
                    "score": {"type": "number"},
                },
                "required": ["start_chunk", "end_chunk", "hook", "score"],
            },
        }
    },
    "required": ["candidates"],
}


class CandidateProposal(BaseModel):
    start_chunk: int
    end_chunk: int
    hook: str = ""
    reason: str = ""
    has_specifics: bool = False
    self_contained: bool = True
    score: float = 0.0


class Clip(BaseModel):
    rank: int
    start: float
    end: float
    duration: float
    start_chunk: int
    end_chunk: int
    topic: Optional[str] = None
    content_type: Optional[str] = None
    hook: str
    reason: str
    score: float
    text: str
    snapped_to_sentence: bool = Field(
        default=False, description="True if a real sentence edge was found to cut on"
    )
    hook_anchored: bool = Field(
        default=False,
        description="True if the start was moved forward to where the hook is spoken",
    )
    url: str = Field(default="", description="public URL once published")


# --------------------------------------------------------------------------- #
# Sentence-boundary detection
# --------------------------------------------------------------------------- #

_SENTENCE_END_CHARS = (".", "?", "!", '."', '?"', '!"')


def _clean_words(words: list[Word]) -> list[Word]:
    """Only words whose timing repair.py did not have to correct.

    Cutting on a repaired timestamp means cutting on a guess.
    """
    return [w for w in words if not w.suspect]


def _sentence_edges(words: list[Word]) -> tuple[list[float], list[float]]:
    """Return (sentence start times, sentence end times).

    Detected from punctuation. Caveat worth knowing: Whisper often returns the
    first stretch of a transcript with no punctuation at all — in the test file
    chunks 0-13 have none. Where punctuation is missing no boundaries are found
    and we fall back to plain word edges, so early clips benefit less from this.
    """
    starts: list[float] = []
    ends: list[float] = []
    expect_start = True

    for w in words:
        if expect_start:
            starts.append(w.start)
            expect_start = False
        if w.word.rstrip().endswith(_SENTENCE_END_CHARS):
            ends.append(w.end)
            expect_start = True

    return starts, ends


def _tokens(text: str) -> list[str]:
    return [t for t in re.sub(r"[^a-z0-9\s']", " ", text.lower()).split() if t]


def _find_hook_start(
    hook: str, words: list[Word], lo_t: float, hi_t: float
) -> Optional[float]:
    """Locate where the hook line is actually spoken, inside [lo_t, hi_t].

    The model reliably identifies the strongest line but does NOT reliably make it
    the first line of its chunk range. In direct mode three of five clips opened on
    filler while the stated hook arrived up to ten seconds later — by which point a
    scrolling viewer is gone.

    So we take the model at its word: find the hook in the transcript and start the
    clip there.

    Matching is by token OVERLAP over a sliding window, not position-by-position,
    because words go missing: the hook "a person who understands that when they
    go..." has to match a transcript reading "a person understands when they go...".
    Strict alignment scores that at 25% and misses entirely.

    Searches ALL words including ones repair.py flagged as suspect. Their TIMING is
    untrustworthy but their TEXT is fine, and we need the text to find the line. One
    hook — "so so the evening and the morning are two times" — failed to match
    because `morning`, `and` and `are` were all suspect, leaving too few words to
    recognise it. The caller moves the result to a trustworthy edge afterwards.
    """
    hook_toks = _tokens(hook)[:8]
    if len(hook_toks) < 3:
        return None
    hook_set = set(hook_toks)

    window_words = [w for w in words if lo_t - 0.5 <= w.start <= hi_t]
    if len(window_words) < 3:
        return None
    flat = []
    for w in window_words:
        t = _tokens(w.word)
        flat.append(t[0] if t else "")

    span = len(hook_toks) + 4
    scores = []
    for i in range(max(1, len(flat) - 2)):
        seen = set(flat[i : i + span])
        scores.append(len(hook_set & seen) / len(hook_set))

    best = max(scores, default=0.0)
    if best < 0.6:
        return None

    tied = [i for i, sc in enumerate(scores) if sc >= best - 1e-9]
    # Among equally good windows, prefer one that BEGINS on a hook word — that
    # aligns the cut with the hook itself rather than a few seconds of lead-in.
    # Sentence snapping runs afterwards and will pull it to the sentence edge.
    on_hook = [i for i in tied if flat[i] in hook_set]
    best_i = (on_hook or tied)[0]
    return window_words[best_i].start


def _nearest(t: float, options: list[float], lo: float, hi: float) -> Optional[float]:
    """Closest option within [t-lo, t+hi], or None."""
    window = [c for c in options if t - lo <= c <= t + hi]
    if not window:
        return None
    return min(window, key=lambda c: abs(c - t))


def _snap_word(t: float, words: list[Word], edge: str, window: float = 0.6) -> float:
    if not words:
        return t
    if edge == "start":
        cands = [w.start for w in words if abs(w.start - t) <= window]
        return min(cands, default=t)
    cands = [w.end for w in words if abs(w.end - t) <= window]
    return max(cands, default=t)


# --------------------------------------------------------------------------- #
# Duration enforcement
# --------------------------------------------------------------------------- #


def _fit_duration(
    proposal: CandidateProposal, transcript: Transcript
) -> Optional[tuple[int, int]]:
    segs = transcript.segments
    lo, hi = proposal.start_chunk, proposal.end_chunk
    if not (0 <= lo <= hi < len(segs)):
        return None

    def dur(a: int, b: int) -> float:
        return segs[b].end - segs[a].start

    # Too long: trim from the end. The hook lives at the start, so protect it.
    while dur(lo, hi) > TARGET_MAX and hi > lo:
        hi -= 1

    # Too short: grow, preferring to add context after the hook.
    while dur(lo, hi) < TARGET_MIN:
        grew = False
        if hi + 1 < len(segs):
            hi += 1
            grew = True
        elif lo > 0:
            lo -= 1
            grew = True
        if not grew:
            break

    d = dur(lo, hi)
    if d < TARGET_MIN or d > TARGET_MAX:
        return None
    return lo, hi


def _overlaps(a: Clip, b: Clip, tolerance: float = 2.0) -> bool:
    return a.start < b.end - tolerance and b.start < a.end - tolerance


# --------------------------------------------------------------------------- #
# Prompt input
# --------------------------------------------------------------------------- #


def _chunk_block(transcript: Transcript) -> str:
    return "\n".join(
        f"[{s.index}] ({s.start:.1f}s) {s.text}" for s in transcript.segments
    )


def _build_user_message(analysis: VideoAnalysis, mode: Mode) -> str:
    total = len(analysis.transcript.segments)
    header = (
        f"Video is {analysis.metadata.duration:.0f} seconds long, with {total} "
        f"transcript chunks numbered 0 to {total - 1}.\n"
    )

    if mode == "direct":
        return header + "\nTRANSCRIPT CHUNKS:\n" + _chunk_block(analysis.transcript)

    lines = [header, "TOPICAL SEGMENTS:"]
    for s in analysis.semantic_segments:
        lines.append(
            f"  chunks {s.start_chunk}-{s.end_chunk} | {s.content_type} | "
            f"{s.topic}: {s.summary}"
        )
    lines += ["", "TRANSCRIPT CHUNKS:", _chunk_block(analysis.transcript)]
    return "\n".join(lines)


def _call_llm(
    analysis: VideoAnalysis, content_type: str, mode: Mode
) -> list[CandidateProposal]:
    playbook = PLAYBOOKS.get(content_type, PLAYBOOKS["generic"])
    template = DIRECT_PROMPT if mode == "direct" else SEGMENTS_PROMPT
    data = complete_json(
        template.format(playbook=playbook),
        _build_user_message(analysis, mode),
        schema=CANDIDATE_SCHEMA,
        list_key="candidates",
        required_field="start_chunk",
    )
    out: list[CandidateProposal] = []
    for c in data.get("candidates", []):
        try:
            out.append(CandidateProposal(**c))
        except Exception:
            continue
    return out


def _text_between(transcript: Transcript, start: float, end: float) -> str:
    """Words actually inside the cut.

    Built from the word list rather than whole chunks, because after hook
    anchoring the clip no longer covers its full starting chunk — reporting the
    chunk text would show words that aren't in the video.
    """
    inside = [w.word for w in transcript.words if start - 0.05 <= w.start < end]
    if inside:
        return " ".join(inside)
    return " ".join(
        s.text.strip() for s in transcript.segments if s.start >= start and s.end <= end
    )


def _label_at(analysis: VideoAnalysis, chunk: int) -> tuple[Optional[str], Optional[str]]:
    """(topic, content_type) of the segment containing this chunk, if any."""
    for s in analysis.semantic_segments:
        if s.start_chunk <= chunk <= s.end_chunk:
            return s.topic, s.content_type
    return None, None


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def select_clips(
    analysis: VideoAnalysis,
    content_type: str = "generic",
    top_n: int = 5,
    mode: Mode = "segments",
) -> list[Clip]:
    if mode == "segments" and not analysis.semantic_segments:
        raise ValueError(
            "mode='segments' needs semantic_segments. Run Step 2 without "
            "--no-segment, or use mode='direct'."
        )

    proposals = _call_llm(analysis, content_type, mode)
    if not proposals:
        return []

    segs = analysis.transcript.segments
    all_words = analysis.transcript.words
    clean = _clean_words(all_words)
    # Sentence edges come from ALL words. Punctuation is a property of the TEXT;
    # `suspect` only means the timing was repaired. Computing edges from clean
    # words alone made "So" invisible in "So there's two times...", so snapping
    # pushed the start forward onto "there's" and clipped the first word.
    sent_starts, sent_ends = _sentence_edges(all_words)

    scored: list[Clip] = []

    for p in proposals:
        fitted = _fit_duration(p, analysis.transcript)
        if fitted is None:
            continue
        lo, hi = fitted

        # Reject anything that starts OR ends in an outro or filler stretch.
        # The old code only checked the start, which let the channel's
        # "subscribe" outro bleed into the tail of the last clip.
        topic, start_type = _label_at(analysis, lo)
        _, end_type = _label_at(analysis, hi)
        if start_type in ("filler", "outro") or end_type in ("filler", "outro"):
            continue

        raw_start, raw_end = segs[lo].start, segs[hi].end

        # Move the start to where the hook is actually spoken, if that's later.
        #
        # Search ALL words, not just trustworthy ones. A suspect word has bad
        # TIMING but perfectly good TEXT, and we need the text to find the line.
        # Matching against clean words only made the hook "so the evening and the
        # morning are two times" unfindable, because `morning`, `and` and `are`
        # were all suspect.
        hook_anchored = False
        hook_t = _find_hook_start(p.hook, all_words, raw_start, raw_end)
        if (
            hook_t is not None
            and raw_start < hook_t <= raw_start + HOOK_MAX_SHIFT
            and raw_end - hook_t >= TARGET_MIN
        ):
            raw_start = lead_in(hook_t, all_words)
            hook_anchored = True

            # Anchoring only ever shortens, so give the time back at the end.
            # Length was fitted before the start moved; a 38s candidate anchored
            # 12s in became 26s with nothing to restore it.
            while (
                hi + 1 < len(segs)
                and segs[hi + 1].end - raw_start <= TARGET_MAX
                and segs[hi].end - raw_start < IDEAL
            ):
                hi += 1
            raw_end = segs[hi].end

        # Prefer a real sentence boundary. Moving the start FORWARD trims a
        # mid-sentence fragment off the opening, which is the single most visible
        # flaw in an automated clip.
        #
        # Once anchored the start must NOT drift backwards — that is how an
        # earlier version undid its own fix, walking back 0.56s from "some people
        # can't think" into "to be the difference some people can't think".
        back = 0.4 if hook_anchored else START_BACK
        snapped = False
        s = _nearest(raw_start, sent_starts, back, START_FORWARD)
        e = _nearest(raw_end, sent_ends, END_WINDOW, END_WINDOW)
        if s is not None:
            snapped = True
            # A sentence start is a word start, so it needs the same lead-in.
            s = lead_in(s, all_words)
        elif hook_anchored:
            s = raw_start          # already padded by lead_in above
        else:
            s = lead_in(_snap_word(raw_start, clean, "start"), all_words)
        if e is not None:
            snapped = True
        else:
            e = _snap_word(raw_end, clean, "end")

        # Snapping must not push the clip outside the acceptable length.
        if e - s < TARGET_MIN or e - s > TARGET_MAX:
            s, e, snapped = raw_start, raw_end, False

        score = p.score
        if not p.self_contained:
            score -= 25
        if not p.has_specifics:
            score -= 15
        # Penalise distance from the ideal length harder than before, and harder
        # for being short than long — a thin clip has nowhere to build.
        gap = (e - s) - IDEAL
        score -= abs(gap) * (1.0 if gap < 0 else 0.5)

        scored.append(
            Clip(
                rank=0,
                start=round(s, 2),
                end=round(e, 2),
                duration=round(e - s, 2),
                start_chunk=lo,
                end_chunk=hi,
                topic=topic,
                content_type=start_type,
                hook=p.hook.strip(),
                reason=p.reason.strip(),
                score=round(score, 1),
                text=_text_between(analysis.transcript, s, e),
                snapped_to_sentence=snapped,
                hook_anchored=hook_anchored,
            )
        )

    # Greedy non-overlapping selection, best first.
    scored.sort(key=lambda c: c.score, reverse=True)
    chosen: list[Clip] = []
    for c in scored:
        if any(_overlaps(c, k) for k in chosen):
            continue
        chosen.append(c)
        if len(chosen) >= top_n:
            break

    for i, c in enumerate(chosen, 1):
        c.rank = i
    return chosen