"""Choose the exact start and end of a clip, as a separate decision.

WHY THIS IS ITS OWN PASS

The selection prompt asks one model to simultaneously judge content, hook
strength, self-containment, AND pick chunk boundaries AND respect a duration
target. That is too many objectives, and the one that loses is completeness —
the clip opens on a fragment or stops before the thought lands.

Vizard published the same finding in July 2026: their old model decided the
moment, the start, the end and the duration in one step, and it prioritised
duration over content completeness. Their fix was a dedicated role whose only
job is where the clip ends, with duration fitted afterwards and a completeness
gate that REJECTS a clip rather than merely scoring it lower. Google's Glance
pipeline is built the same way — transcribe, then a separate model pass to
identify optimal start and end from word-level timestamps.

So the order here is deliberately:

    1. enumerate real candidate edges from the transcript
    2. a model picks the best END first, then the best START
    3. duration is fitted around that choice, never driving it
    4. completeness gate rejects anything still opening or closing mid-thought

WHY A MODEL AND NOT A HEURISTIC

Because the transcript lies in ways no rule can untangle. A real example from
the test file: repair.py sorts words by timestamp, and where two timestamps
overlap it can reverse them. The speech "Because they are different, right?
That's the joy of change" came out as "are / right? / different, / That's",
which makes the fragment "different," look like a sentence start and the clip
opened on it. A model reading the candidate options picks "That's the joy of
change" without difficulty. Punctuation heuristics cannot.

The DSP pass in boundary.py still runs afterwards. This module chooses WHICH
WORD to open on; that one finds the exact acoustic instant beside it.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from .llm import complete_json
from .models import Transcript, Word

# How far either side of the proposed edge to offer alternatives.
START_WINDOW = 12.0
END_WINDOW = 15.0
MAX_OPTIONS = 8

_END_MARKS = (".", "?", "!")


class EdgeChoice(BaseModel):
    start_option: int
    end_option: int
    complete: bool = True
    why: str = ""


SCHEMA = {
    "type": "object",
    "properties": {
        "start_option": {"type": "integer"},
        "end_option": {"type": "integer"},
        "complete": {"type": "boolean"},
        "why": {"type": "string"},
    },
    "required": ["start_option", "end_option", "complete"],
}


SYSTEM_PROMPT = """You are a video editor choosing exactly where a short clip
starts and ends. The content has already been chosen. Your only job is the edges.

You get numbered START options and numbered END options. Each shows the words
that would follow (for a start) or precede (for an end) that cut.

Pick the END first and treat it as the priority. The right end is where the
thought actually lands: the sentence that concludes the argument, the moment the
story pays off, the line someone would quote. Never end on a conjunction, a
trailing "and", "but", "because", "so", or halfway through an idea. A clip that
stops before the point arrives is worthless no matter how good the opening was.

Then pick the START. The right start is a complete grammatical opening that
makes sense to someone who has seen nothing before it. Reject options that open
on a fragment ("different, That's the joy of..."), on a bare acknowledgement
("Right? So..."), or mid-sentence. A couple of seconds of extra context is
better than an opening the viewer cannot parse.

Be aware the transcript is imperfect: speech-to-text sometimes reverses two
adjacent words or misplaces punctuation, so an option may look like a sentence
start when it is really a fragment. Judge by whether the WORDS read as a
sensible opening, not by where the punctuation sits.

Set "complete": false if NO combination available gives both a clean opening and
a landed ending. Saying so is more useful than picking the least bad pair.

Return ONLY JSON:
{"start_option": int, "end_option": int, "complete": bool, "why": "one line"}"""


def _sentence_starts(words: list[Word]) -> list[int]:
    """Indices of words that plausibly begin a sentence.

    Two signals, because punctuation alone is not enough. A word following a
    full stop obviously starts a sentence. But so does a capitalised word in
    mid-sequence: the ASR capitalised it because it heard a sentence beginning
    there, even if it only wrote a comma before it.

    That second signal is what surfaces "That's the joy of change" as an option
    in "…are different, That's the joy of change" — the case where a clip
    otherwise opens on the orphan fragment "different,".

    Proper nouns get offered too. That is fine: an extra candidate costs
    nothing because the model chooses between them.
    """
    if not words:
        return []
    out = [0]
    for i in range(1, len(words)):
        prev = words[i - 1].word.rstrip()
        tok = words[i].word.strip()
        after_stop = prev.endswith(_END_MARKS)
        capitalised = (
            bool(tok)
            and tok[0].isupper()
            and tok.lower() not in ("i", "i'm", "i'll", "i've", "i'd")
        )
        if after_stop or capitalised:
            out.append(i)
    return out


def _sentence_ends(words: list[Word]) -> list[int]:
    return [
        i for i, w in enumerate(words) if w.word.rstrip().endswith(_END_MARKS)
    ]


def _text(words: list[Word], lo: int, hi: int, limit: int = 130) -> str:
    s = " ".join(w.word for w in words[lo : hi + 1])
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _start_options(
    words: list[Word], proposed: float
) -> list[tuple[int, float]]:
    """Candidate opening words near the proposed start."""
    cands = [
        i
        for i in _sentence_starts(words)
        if proposed - 3.0 <= words[i].start <= proposed + START_WINDOW
    ]
    if not cands:
        cands = [
            i
            for i, w in enumerate(words)
            if proposed - 0.5 <= w.start <= proposed + 2.0
        ]
    # Keep the earliest few — later options shorten the clip.
    return [(i, words[i].start) for i in cands[:MAX_OPTIONS]]


def _end_options(words: list[Word], proposed: float) -> list[tuple[int, float]]:
    """Candidate closing words near the proposed end.

    Offered both before and after the proposal, because the natural ending is
    frequently LATER than a duration-constrained guess — that is the whole
    failure mode this pass exists to fix.
    """
    cands = [
        i
        for i in _sentence_ends(words)
        if proposed - END_WINDOW <= words[i].end <= proposed + END_WINDOW
    ]
    if not cands:
        cands = [
            i
            for i, w in enumerate(words)
            if proposed - 2.0 <= w.end <= proposed + 2.0
        ]
    if len(cands) > MAX_OPTIONS:
        # Keep a spread rather than only the nearest, so a genuinely later
        # ending stays reachable.
        step = len(cands) / MAX_OPTIONS
        cands = [cands[int(i * step)] for i in range(MAX_OPTIONS)]
    return [(i, words[i].end) for i in cands]


def optimise_edges(
    transcript: Transcript,
    start: float,
    end: float,
    *,
    min_s: float,
    max_s: float,
    hook: str = "",
    verbose: bool = True,
) -> tuple[float, float, bool, str]:
    """Pick better edges for one clip. Returns (start, end, complete, why).

    Falls back to the input on any failure — this improves a clip that already
    works, so it must never be the reason a run fails.
    """
    words = transcript.words
    if not words:
        return start, end, True, "no word list"

    s_opts = _start_options(words, start)
    e_opts = _end_options(words, end)
    if not s_opts or not e_opts:
        return start, end, True, "no alternatives found"

    lines = ["START options (words that would open the clip):"]
    for n, (i, t) in enumerate(s_opts):
        lines.append(f"  [{n}] {t:.1f}s  {_text(words, i, i + 16)}")
    lines.append("")
    lines.append("END options (words that would close the clip):")
    for n, (i, t) in enumerate(e_opts):
        lines.append(f"  [{n}] {t:.1f}s  …{_text(words, max(0, i - 16), i)}")
    lines.append("")
    lines.append(
        f"Target length {min_s:.0f}-{max_s:.0f}s. "
        f"Pick the end for completeness first, then a start that fits."
    )
    if hook:
        lines.append(f'The chosen content was described as: "{hook[:120]}"')

    try:
        data = complete_json(
            SYSTEM_PROMPT,
            "\n".join(lines),
            schema=SCHEMA,
            list_key="choice",
            required_field="start_option",
            temperature=0.0,
        )
        choice = EdgeChoice(**data)
    except Exception as e:
        if verbose:
            print(f"      edges: pass failed, keeping originals ({e})")
        return start, end, True, "edge pass failed"

    if not (0 <= choice.start_option < len(s_opts)):
        return start, end, True, "start option out of range"
    if not (0 <= choice.end_option < len(e_opts)):
        return start, end, True, "end option out of range"

    new_s = s_opts[choice.start_option][1]
    new_e = e_opts[choice.end_option][1]

    if new_e - new_s < 5.0:
        return start, end, True, "chosen edges collapse the clip"

    # Duration is fitted AROUND the chosen ending, never allowed to override it.
    if new_e - new_s > max_s:
        for i, t in reversed(s_opts):
            if new_e - t <= max_s and new_e - t >= min_s:
                new_s = t
                break
        else:
            new_s = new_e - max_s
    elif new_e - new_s < min_s:
        for i, t in s_opts:
            if new_e - t >= min_s:
                new_s = t
                break
        else:
            # Cannot reach the floor without losing the ending. Keep the ending.
            new_s = max(0.0, new_e - min_s)

    return round(new_s, 3), round(new_e, 3), choice.complete, choice.why