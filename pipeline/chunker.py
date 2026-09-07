"""Split long audio into upload-sized pieces, transcribe each, stitch back.

Groq caps uploads at 25 MB. At 16 kHz mono PCM that is about 12 minutes, which
is fine for testing and useless for real client footage — a 2-hour recording is
roughly 230 MB.

The naive fix is to cut every N minutes, but a fixed cut lands mid-word and both
sides of the split lose a word. So we cut at silence instead: find the quiet
gaps with ffmpeg, then choose the gap nearest each target boundary.

Timestamps from each piece are relative to that piece, so every one is offset
back to absolute time before merging. Get that wrong and the whole transcript is
silently misaligned.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from .models import Transcript, TranscriptSegment, Word

# 25 MB cap, minus headroom for the multipart envelope.
MAX_BYTES = 23_000_000


def safe_chunk_seconds(audio: Path, duration: float) -> float:
    """How many seconds fit in one upload, measured from this file.

    An earlier version assumed 32,000 bytes/sec — correct for 16 kHz mono PCM
    and wrong by 4x for the 64 kbps MP3 the pipeline now extracts. Hardcoding a
    byte rate means the chunk length silently stops matching the format, so we
    measure the file we actually have.
    """
    if duration <= 0:
        return 600.0
    rate = audio.stat().st_size / duration
    # 85% of the theoretical maximum. A chunk cut at a silence lands a little
    # either side of the target, and `-c copy` rounds to a frame boundary, so a
    # chunk sized to exactly fill the limit can come back marginally over it.
    return max(120.0, min(MAX_BYTES * 0.85 / rate, 3000.0))


def _run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr[-1500:]}")
    return r.stderr + r.stdout          # silencedetect writes to stderr


def find_silences(
    audio: Path, noise_db: float = -32.0, min_dur: float = 0.45
) -> list[float]:
    """Midpoints of silent stretches, in seconds.

    -32 dB and 0.45s are deliberately loose. We only need a handful of usable
    boundaries, not every pause, and a tight threshold on a noisy Zoom recording
    finds nothing at all.
    """
    out = _run([
        "ffmpeg", "-i", str(audio), "-af",
        f"silencedetect=noise={noise_db}dB:d={min_dur}",
        "-f", "null", "-", "-loglevel", "info",
    ])
    starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", out)]
    ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", out)]
    return [round((s + e) / 2, 3) for s, e in zip(starts, ends)]


def plan_cuts(duration: float, silences: list[float],
              chunk_sec: float = 700.0) -> list[tuple[float, float]]:
    """Choose chunk boundaries at silences near each target point."""
    if duration <= chunk_sec:
        return [(0.0, duration)]

    bounds = [0.0]
    while bounds[-1] + chunk_sec < duration:
        target = bounds[-1] + chunk_sec
        # Only consider silences comfortably after the previous boundary, or a
        # cluster of pauses can produce a chunk two seconds long.
        usable = [s for s in silences if bounds[-1] + 60 < s <= target]
        bounds.append(max(usable) if usable else target)
    bounds.append(duration)
    return list(zip(bounds, bounds[1:]))


def split_audio(audio: Path, cuts: list[tuple[float, float]], out_dir: Path) -> list[Path]:
    """Cut the audio at the planned boundaries, keeping the source format.

    An earlier version wrote every chunk as 16 kHz PCM regardless of what came
    in. That silently undid the compression: a 37 MB MP3 split into two pieces
    produced a 92 MB WAV chunk — bigger than the whole original — and Groq
    rejected it. The chunk length is calculated from the source bitrate, so the
    chunks have to keep that bitrate or the arithmetic means nothing.

    `-c copy` also avoids re-encoding, which is faster and lossless. It can only
    cut on frame boundaries, so a chunk may land a few milliseconds off the
    requested time; that is well inside the silence we chose to cut in.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = audio.suffix or ".mp3"
    paths = []
    for i, (start, end) in enumerate(cuts):
        p = out_dir / f"chunk_{i:03d}{ext}"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(audio),
            "-ss", str(start), "-to", str(end),
            "-c", "copy",
            "-loglevel", "error", str(p),
        ], check=True)
        mb = p.stat().st_size / 1e6
        if mb * 1e6 > MAX_BYTES:
            # Catch it here rather than after the upload is refused. Cutting the
            # target in half and re-planning is cheaper than a failed API call.
            raise RuntimeError(
                f"chunk {i} came out {mb:.0f} MB, over the {MAX_BYTES/1e6:.0f} MB "
                f"limit. The source is {audio.suffix}; if that is .wav the "
                f"pipeline extracted uncompressed audio for a hosted backend."
            )
        paths.append(p)
    return paths


def merge(parts: list[tuple[Transcript, float]]) -> Transcript:
    """Stitch chunk transcripts into one, shifting each by its start offset."""
    segments: list[TranscriptSegment] = []
    words: list[Word] = []
    language = "en"

    for tr, offset in parts:
        language = tr.language or language
        for s in tr.segments:
            segments.append(TranscriptSegment(
                index=len(segments),
                start=round(s.start + offset, 3),
                end=round(s.end + offset, 3),
                text=s.text,
                speaker=s.speaker,
            ))
        for w in tr.words:
            words.append(Word(
                word=w.word,
                start=round(w.start + offset, 3),
                end=round(w.end + offset, 3),
                score=w.score,
                suspect=w.suspect,
            ))

    return Transcript(language=language, segments=segments, words=words)


def transcribe_long(audio: Path, transcribe_fn, verbose: bool = True) -> Transcript:
    """Transcribe an audio file of any length.

    `transcribe_fn` takes a Path and returns a Transcript — pass the normal
    single-file transcriber.
    """
    size_mb = audio.stat().st_size / 1e6
    if size_mb <= MAX_BYTES / 1e6:
        return transcribe_fn(audio)

    dur = float(subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(audio)],
        capture_output=True, text=True, check=True,
    ).stdout.strip())

    if verbose:
        print(f"      {size_mb:.0f} MB / {dur/60:.0f} min — splitting for upload")

    silences = find_silences(audio)
    chunk_sec = safe_chunk_seconds(audio, dur)
    cuts = plan_cuts(dur, silences, chunk_sec)
    if verbose:
        print(f"      {len(silences)} silences found, {len(cuts)} chunks planned "
              f"(~{chunk_sec/60:.0f} min each at this bitrate)")

    tmp = Path(tempfile.mkdtemp(prefix="chunks_"))
    parts = []
    for i, (path, (start, end)) in enumerate(zip(split_audio(audio, cuts, tmp), cuts)):
        if verbose:
            print(f"      chunk {i+1}/{len(cuts)}  {start/60:.1f}-{end/60:.1f} min")
        parts.append((transcribe_fn(path), start))
        path.unlink(missing_ok=True)

    merged = merge(parts)
    if verbose:
        print(f"      merged: {len(merged.segments)} chunks, {len(merged.words)} words")
    return merged