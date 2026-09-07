"""Repair Whisper's word timestamps so they form a valid sequence.

Whisper derives word timings from cross-attention weights, not from acoustic
alignment. The result is usually close but not ordered: words overlap, start
before the previous word ended, occasionally start before the previous word
*started*, and sometimes get durations of 20ms or 5s. This is inherent to the
method and happens on large-v3 and Groq exactly as it does on smaller models.

Step 3 snaps clip boundaries to word edges, and that search assumes the list is
monotonic. So we enforce it here rather than letting invalid data propagate.

Nothing is deleted. Words that needed correcting are flagged `suspect=True` so
downstream code can refuse to use them as snap targets and walk to the nearest
clean boundary instead.
"""

from __future__ import annotations

from .models import Word

MIN_DURATION = 0.04   # below this, the timing is noise
MAX_DURATION = 1.20   # no English word takes longer; Whisper stretched it
                      # across silence or music


def repair_words(words: list[Word]) -> tuple[list[Word], dict[str, int]]:
    """Return monotonic words plus a count of what was corrected."""
    if not words:
        return [], {}

    stats = {"overlap": 0, "too_short": 0, "too_long": 0, "reordered": 0}

    # Sort by start time. Stable, so genuine ties keep transcript order.
    ordered = sorted(words, key=lambda w: (w.start, w.end))
    if [id(w) for w in ordered] != [id(w) for w in words]:
        stats["reordered"] = sum(
            1 for a, b in zip(words, ordered) if id(a) != id(b)
        )

    repaired: list[Word] = []
    prev_end = 0.0

    for w in ordered:
        start, end = w.start, w.end
        suspect = False

        # A word stretched across silence: trust the end, pull the start in.
        if end - start > MAX_DURATION:
            start = end - MAX_DURATION
            suspect = True
            stats["too_long"] += 1

        # Overlap with the previous word: push the start forward.
        if start < prev_end:
            start = prev_end
            suspect = True
            stats["overlap"] += 1

        # Collapsed or inverted after clamping: give it a minimum slot.
        if end - start < MIN_DURATION:
            end = start + MIN_DURATION
            suspect = True
            stats["too_short"] += 1

        repaired.append(
            Word(
                word=w.word,
                start=round(start, 3),
                end=round(end, 3),
                score=w.score,
                suspect=suspect,
            )
        )
        prev_end = end

    return repaired, {k: v for k, v in stats.items() if v}


def verify_monotonic(words: list[Word]) -> list[str]:
    """Assert the invariant Step 3 depends on. Returns violation messages."""
    problems = []
    for i in range(1, len(words)):
        prev, cur = words[i - 1], words[i]
        if cur.start < prev.end:
            problems.append(
                f"[{i}] {cur.word!r} starts {cur.start} before "
                f"{prev.word!r} ends {prev.end}"
            )
        if cur.end <= cur.start:
            problems.append(f"[{i}] {cur.word!r} has non-positive duration")
    return problems
