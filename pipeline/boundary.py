"""Move a cut point to the real acoustic gap, measured from the audio.

The problem this solves, stated precisely: an ASR word timestamp tells you WHICH
word ended where, not the exact instant. In connected speech the articulators
don't reset between words (coarticulation), so the true quiet gap sits tens of
milliseconds from the reported boundary — sometimes earlier, sometimes later.

That is why padding cannot fix it. A fixed lead-in helps when the timestamp is
late and hurts when it is early, which is exactly what we saw: clip #2 needed to
move 0.45s earlier, while three others were already correct and got worse.

Three bounded passes, no model, no GPU:

  1. Energy minimum   search +/-60ms in 10ms RMS frames for the quietest instant
  2. Zero crossing    snap +/-5ms to a waveform sign change, so the cut itself
                      introduces no click
  3. Hard clamp       reject any candidate that would cross into a neighbouring
                      word, and fall back to the original boundary

Step 3 is the load-bearing guarantee. Refinement can move a cut into real
silence or do nothing at all; it can never eat into a word we meant to keep.

Technique from dougcalobrisi/erm's refine.py, corroborated by librosa's own
rationale for onset_detect(backtrack=True): walk back from a rough estimate to
the nearest true local minimum in energy rather than trusting the estimate.

Note this is NOT general silence detection. Snapping cuts to silence across a
whole file drags deliberate boundaries into filler words and awkward pauses.
This search is bounded to 60ms — it refines a boundary we already chose.

Audio is decoded via an ffmpeg subprocess rather than librosa: ffmpeg is already
a hard dependency, and librosa pulls a large scipy chain we need nothing else
from.
"""

from __future__ import annotations

import array
import math
import subprocess
from pathlib import Path
from typing import Optional

SR = 16000              # analysis sample rate; plenty for locating a gap
FRAME_MS = 10.0         # RMS frame length
SEARCH_MS = 60.0        # how far either side to look for the quiet point
ZERO_SNAP_MS = 5.0      # how far to nudge onto a zero crossing
PAD_MS = 250.0          # extra audio to decode either side of the window


def _decode(
    media: Path, start: float, duration: float
) -> Optional[array.array]:
    """Mono 16-bit PCM for a slice of the media, via ffmpeg."""
    if duration <= 0:
        return None
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "quiet",
            "-ss", f"{max(0.0, start):.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(media),
            "-vn", "-ac", "1", "-ar", str(SR),
            "-f", "s16le", "-",
        ],
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    samples = array.array("h")
    n = len(proc.stdout) - (len(proc.stdout) % 2)
    samples.frombytes(proc.stdout[:n])
    return samples or None


def _rms_envelope(samples: array.array) -> tuple[list[float], int]:
    """Short-time RMS. Returns (frame energies, samples per frame)."""
    per = max(1, int(SR * FRAME_MS / 1000.0))
    out: list[float] = []
    for i in range(0, len(samples) - per + 1, per):
        total = 0
        for s in samples[i : i + per]:
            total += s * s
        out.append(math.sqrt(total / per))
    return out, per


def _snap_to_zero_crossing(samples: array.array, idx: int) -> int:
    """Nearest sign change within ZERO_SNAP_MS of idx."""
    span = max(1, int(SR * ZERO_SNAP_MS / 1000.0))
    lo = max(1, idx - span)
    hi = min(len(samples) - 1, idx + span)
    best, best_d = idx, span + 1
    for i in range(lo, hi):
        if (samples[i - 1] < 0) != (samples[i] < 0):
            d = abs(i - idx)
            if d < best_d:
                best, best_d = i, d
    return best


def refine(
    media: Path,
    t: float,
    *,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> tuple[float, bool]:
    """Refine a cut point. Returns (time, moved).

    `lower` and `upper` clamp the result — pass the neighbouring words' own
    timestamps. Without them the search is still bounded to +/-60ms, but the
    clamp is what makes this safe to leave on by default.

    Any failure returns the input unchanged. This is a quality improvement on a
    cut that already works, never a correctness requirement.
    """
    try:
        win_start = max(0.0, t - (SEARCH_MS + PAD_MS) / 1000.0)
        win_len = 2 * (SEARCH_MS + PAD_MS) / 1000.0
        samples = _decode(media, win_start, win_len)
        if samples is None or len(samples) < SR * 0.05:
            return t, False

        env, per = _rms_envelope(samples)
        if len(env) < 3:
            return t, False

        centre = int((t - win_start) * SR)
        span = int(SR * SEARCH_MS / 1000.0)

        f_lo = max(0, (centre - span) // per)
        f_hi = min(len(env) - 1, (centre + span) // per)
        if f_hi <= f_lo:
            return t, False

        quietest = min(range(f_lo, f_hi + 1), key=lambda i: env[i])
        # Middle of the quietest frame, then onto a zero crossing.
        idx = int((quietest + 0.5) * per)
        idx = _snap_to_zero_crossing(samples, idx)
        candidate = win_start + idx / SR

        if lower is not None and candidate < lower:
            return t, False
        if upper is not None and candidate > upper:
            return t, False
        if abs(candidate - t) < 0.002:
            return t, False
        return round(candidate, 3), True
    except Exception:
        return t, False


def neighbours(
    words: list, t: float, edge: str
) -> tuple[Optional[float], Optional[float]]:
    """Clamp bounds for a cut at `t`, from the surrounding words.

    For a clip START the cut may sit anywhere between the end of the previous
    word and the start of the word we are keeping — it must not eat into that
    word's onset. For a clip END, between the end of the last kept word and the
    start of the next one.
    """
    if not words:
        return None, None
    if edge == "start":
        prev_end = max((w.end for w in words if w.end <= t + 0.01), default=None)
        nxt_start = min((w.start for w in words if w.start >= t - 0.01), default=None)
        return prev_end, nxt_start
    prev_end = max((w.end for w in words if w.end <= t + 0.01), default=None)
    nxt_start = min((w.start for w in words if w.start >= t - 0.01), default=None)
    return prev_end, nxt_start