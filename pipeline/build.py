"""Step 2 orchestrator: video.mp4 -> VideoAnalysis JSON."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from .config import config
from .models import VideoAnalysis
from .chunker import transcribe_long
from .probe import extract_audio, extract_metadata
from .llm import describe_backend
from .punctuate import apply_to_words, repunctuate
from .segment import segment_transcript
from .transcribe import transcribe


def analyse_video(
    video_path: Path,
    source: str = "local",
    skip_segmentation: bool = False,
) -> VideoAnalysis:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    t0 = time.time()

    print(f"[1/3] probing {video_path.name}")
    metadata = extract_metadata(video_path, source=source)
    print(f"      {metadata.duration:.1f}s | {metadata.width}x{metadata.height}")

    tmpdir = Path(tempfile.mkdtemp(prefix="vidpipe_"))
    # Hosted backends upload the file, so compression decides how long a video
    # fits in one call: 25 MB is 12 minutes of PCM but 52 minutes of 64k MP3.
    # Local backends read from disk with no limit, so PCM is fine there.
    ext = "wav" if config.asr_backend == "faster_whisper" else "mp3"
    audio_path = tmpdir / f"{metadata.video_id}.{ext}"
    if config.keep_audio:
        audio_path = Path(config.output_dir) / f"{metadata.video_id}.{ext}"

    print(f"[2/3] transcribing via {config.asr_backend}")
    extract_audio(video_path, audio_path)
    size_mb = audio_path.stat().st_size / 1e6
    print(f"      audio: {size_mb:.1f} MB ({ext})")
    # transcribe_long is a passthrough when the file already fits.
    transcript = transcribe_long(audio_path, transcribe)
    metadata.language = transcript.language

    if config.repunctuate and transcript.words:
        transcript, n = repunctuate(transcript)
        if n:
            w = apply_to_words(transcript)
            print(f"      punctuation: pushed onto {w} words")

    print(
        f"      {len(transcript.segments)} chunks, "
        f"{len(transcript.words)} words, lang={transcript.language}"
    )

    if skip_segmentation:
        print("[3/3] segmentation skipped")
        segments = []
    else:
        print(f"[3/3] segmenting via {describe_backend(config.segment_backend)}")
        segments = segment_transcript(transcript)
        print(f"      {len(segments)} semantic segments")

    if not config.keep_audio and audio_path.exists():
        audio_path.unlink()

    print(f"done in {time.time() - t0:.1f}s")

    return VideoAnalysis(
        metadata=metadata,
        transcript=transcript,
        semantic_segments=segments,
    )


def save(analysis: VideoAnalysis, out_dir: Path | str | None = None) -> Path:
    out_dir = Path(out_dir or config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{analysis.metadata.video_id}.json"
    out_path.write_text(
        json.dumps(analysis.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path