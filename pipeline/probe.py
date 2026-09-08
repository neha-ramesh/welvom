"""Step 2a — metadata + audio extraction.

Both are thin wrappers over ffmpeg/ffprobe. No AI involved, and no reason for
any: ffprobe already knows the duration, resolution and fps exactly.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .models import Metadata


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{result.stderr[-2000:]}")
    return result.stdout


def video_id_for(path: Path) -> str:
    """Stable ID from file content, not filename.

    Uses the first 8 MB rather than the whole file — hashing a 2 GB video to
    get an ID is a waste of I/O, and 8 MB of a distinct video is plenty.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(8 * 1024 * 1024))
    h.update(str(path.stat().st_size).encode())
    return h.hexdigest()[:12]


def _metadata_via_ffmpeg(video_path: Path) -> dict:
    """Duration and dimensions parsed from ffmpeg's own output.

    ffprobe is the right tool and is not always present: the pip-installed
    ffmpeg that makes deployment possible without apt ships the encoder only.
    Rather than make ffprobe a hard requirement — and lose the whole clip tab on
    any host where apt is unavailable — read the same facts from what ffmpeg
    prints to stderr when handed a file with no output specified.

    Less precise than ffprobe's JSON, and that is fine: duration to a hundredth
    of a second is ample for cutting clips.
    """
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(video_path)],
        capture_output=True, text=True,
    )
    err = proc.stderr

    out: dict = {}
    m = re.search(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)", err)
    if m:
        h, mi, sec = int(m.group(1)), int(m.group(2)), float(m.group(3))
        out["duration"] = h * 3600 + mi * 60 + sec

    m = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})", err)
    if m:
        out["width"], out["height"] = int(m.group(1)), int(m.group(2))

    m = re.search(r"(\d+(?:\.\d+)?)\s*fps", err)
    if m:
        out["fps"] = round(float(m.group(1)), 3)

    return out


def extract_metadata(video_path: Path, source: str = "local") -> Metadata:
    if not shutil.which("ffprobe"):
        got = _metadata_via_ffmpeg(video_path)
        if not got.get("duration"):
            raise RuntimeError(
                f"Could not read {video_path.name}. Neither ffprobe nor ffmpeg "
                f"could parse it."
            )
        return Metadata(
            video_id=video_id_for(video_path),
            source=source,
            duration=float(got["duration"]),
            title=video_path.stem,
            width=got.get("width"),
            height=got.get("height"),
            fps=got.get("fps"),
            processed_at=datetime.now(timezone.utc).isoformat(),
        )

    raw = _run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
    )
    probe = json.loads(raw)

    fmt = probe.get("format", {})
    video_stream = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
        {},
    )

    fps = None
    rate = video_stream.get("r_frame_rate", "")
    if "/" in rate:
        num, den = rate.split("/")
        if float(den) != 0:
            fps = round(float(num) / float(den), 3)

    return Metadata(
        video_id=video_id_for(video_path),
        source=source,
        duration=float(fmt.get("duration", 0.0)),
        title=fmt.get("tags", {}).get("title") or video_path.stem,
        width=video_stream.get("width"),
        height=video_stream.get("height"),
        fps=fps,
        processed_at=datetime.now(timezone.utc).isoformat(),
    )


def extract_audio(
    video_path: Path, out_path: Path, *, bitrate: str = "64k"
) -> Path:
    """Extract 16 kHz mono audio. Format follows the file extension.

    WAV vs MP3 matters more than it looks. Groq caps uploads at 25 MB, and
    16 kHz mono PCM runs 32,000 bytes per second — so a WAV hits the cap at
    about 12 minutes. The same 25 MB of 64 kbps mono MP3 holds 52 minutes.

    64 kbps mono is well above what speech recognition needs; Whisper resamples
    to 16 kHz internally regardless. Local backends get PCM because there is no
    upload limit and decoding is marginally faster.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()

    if ext == ".wav":
        codec = ["-c:a", "pcm_s16le"]
    elif ext == ".flac":
        codec = ["-c:a", "flac", "-compression_level", "8"]
    else:  # mp3 and anything else
        codec = ["-c:a", "libmp3lame", "-b:a", bitrate]

    _run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",                  # drop video
            "-ac", "1",             # mono
            "-ar", "16000",         # 16 kHz
            *codec,
            "-loglevel", "error",
            str(out_path),
        ]
    )
    return out_path