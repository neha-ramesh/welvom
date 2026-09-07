"""Step 2b — transcription.

Two backends behind one function. Both return the same Transcript shape, so the
rest of the pipeline doesn't know or care which ran.

  faster_whisper : local, free, needs a GPU to be quick. large-v3.
  groq           : hosted, ~$0.04/audio-hour, no GPU, 25 MB upload cap.
  sarvam         : hosted, Indian-accent English and Indic languages. Good
                   punctuation and speaker labels, but NO word timestamps and a
                   30-second cap on the simple endpoint.

Both are asked for word-level timestamps. Segment timestamps alone are not
enough — Step 3 snaps clip boundaries to word edges so clips don't start
mid-syllable, which is the most obvious tell of an automated edit.
"""

from __future__ import annotations

from pathlib import Path

from .config import config
from .models import Transcript, TranscriptSegment, Word
from .repair import repair_words


# Whisper sometimes returns a language name instead of an ISO code.
_LANG_ALIASES = {
    "english": "en", "hindi": "hi", "spanish": "es", "french": "fr",
    "german": "de", "tamil": "ta", "telugu": "te", "kannada": "kn",
    "portuguese": "pt", "italian": "it", "japanese": "ja", "korean": "ko",
    "chinese": "zh", "arabic": "ar", "russian": "ru", "dutch": "nl",
}


def _normalise_language(lang: str) -> str:
    key = (lang or "").strip().lower()
    return _LANG_ALIASES.get(key, key[:5] or "en")


def transcribe(audio_path: Path) -> Transcript:
    backend = config.asr_backend
    if backend == "faster_whisper":
        result = _transcribe_faster_whisper(audio_path)
    elif backend == "groq":
        result = _transcribe_groq(audio_path)
    elif backend == "sarvam":
        result = _transcribe_sarvam(audio_path)
    else:
        raise ValueError(f"Unknown ASR_BACKEND: {backend!r}")

    result.language = _normalise_language(result.language)

    fixed, stats = repair_words(result.words)
    result.words = fixed
    if stats:
        detail = ", ".join(f"{k}={v}" for k, v in stats.items())
        print(f"      repaired word timings: {detail}")

    return result


# --------------------------------------------------------------------------- #
# Local: faster-whisper
# --------------------------------------------------------------------------- #

_model_cache: dict[str, object] = {}


def _get_local_model():
    """Cached so batch runs don't reload ~3 GB of weights per video."""
    key = f"{config.whisper_model}:{config.resolved_device()}"
    if key not in _model_cache:
        from faster_whisper import WhisperModel

        print(
            f"  loading {config.whisper_model} on {config.resolved_device()} "
            f"({config.resolved_compute_type()})..."
        )
        _model_cache[key] = WhisperModel(
            config.whisper_model,
            device=config.resolved_device(),
            compute_type=config.resolved_compute_type(),
        )
    return _model_cache[key]


def _transcribe_faster_whisper(audio_path: Path) -> Transcript:
    model = _get_local_model()

    segments_iter, info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        # VAD trims silence, which both speeds things up and stops Whisper
        # hallucinating text into quiet stretches — a real failure mode.
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        condition_on_previous_text=False,  # reduces repetition loops
    )

    segments: list[TranscriptSegment] = []
    words: list[Word] = []

    for i, seg in enumerate(segments_iter):
        text = seg.text.strip()
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                index=len(segments),
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=text,
            )
        )
        for w in seg.words or []:
            words.append(
                Word(
                    word=w.word.strip(),
                    start=round(w.start, 3),
                    end=round(w.end, 3),
                    score=round(w.probability, 4) if w.probability else None,
                )
            )

    return Transcript(language=info.language, segments=segments, words=words)


# --------------------------------------------------------------------------- #
# Hosted: Groq
# --------------------------------------------------------------------------- #


def _transcribe_groq(audio_path: Path) -> Transcript:
    from groq import Groq

    if not config.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    size_mb = audio_path.stat().st_size / 1_000_000
    if size_mb > 24:
        raise RuntimeError(
            f"Audio is {size_mb:.1f} MB; Groq caps uploads at 25 MB. "
            "Chunk the audio or switch ASR_BACKEND=faster_whisper. "
            "(16 kHz mono PCM is ~1.9 MB/min, so ~12 min is the practical limit.)"
        )

    client = Groq(api_key=config.groq_api_key)
    with audio_path.open("rb") as f:
        result = client.audio.transcriptions.create(
            file=(audio_path.name, f.read()),
            model=config.groq_model,
            response_format="verbose_json",
            # Whisper continues the style of whatever it's primed with, so a
            # punctuated example makes it punctuate its own output. Without this
            # it returned the first ~100 seconds of the test file with no full
            # stops at all, which left clip cutting nothing to snap to.
            #
            # It is a STYLE hint, not an instruction — Whisper can ignore it, and
            # it can occasionally leak words into the transcript. Keep it generic
            # and rely on punctuate.py to catch what slips through.
            prompt=(
                "Hello, welcome back. Today we're talking about something "
                "important. Why does it matter? Well, here's the thing. It "
                "changes everything, right?"
            ),
            timestamp_granularities=["word", "segment"],
        )

    data = result if isinstance(result, dict) else result.model_dump()

    segments = [
        TranscriptSegment(
            index=i,
            start=round(float(s["start"]), 3),
            end=round(float(s["end"]), 3),
            text=s["text"].strip(),
        )
        for i, s in enumerate(data.get("segments", []))
        if s.get("text", "").strip()
    ]

    words = [
        Word(
            word=w["word"].strip(),
            start=round(float(w["start"]), 3),
            end=round(float(w["end"]), 3),
        )
        for w in data.get("words", [])
    ]

    return Transcript(
        language=data.get("language", "en"),
        segments=segments,
        words=words,
    )


# --------------------------------------------------------------------------- #
# Hosted: Sarvam (Indian-accent English and Indic languages)
# --------------------------------------------------------------------------- #


def _transcribe_sarvam(audio_path: Path) -> Transcript:
    """Saarik/Saaras via Sarvam.

    Worth knowing before you switch to this:

    - It does NOT return word-level timestamps. Sentence or phrase level only.
      That is fine for choosing clips and for snapping to sentence starts, but
      the word list will be empty, so anything relying on word edges degrades.
    - Its punctuation is good, which is the main reason to reach for it here.
    - It DOES do speaker diarization (an extra charge), which Groq does not.
    - The simple endpoint caps at 30 seconds of audio; longer files need their
      batch API, which is a different flow entirely.

    Tested side by side with Groq on Indian-accented English: both made the same
    transcription errors, so accuracy is not the reason to choose it.
    """
    from sarvamai import SarvamAI

    if not config.sarvam_api_key:
        raise RuntimeError("SARVAM_API_KEY is not set")

    client = SarvamAI(api_subscription_key=config.sarvam_api_key)
    with audio_path.open("rb") as f:
        r = client.speech_to_text.transcribe(
            file=f,
            model=config.sarvam_model,
            language_code=config.sarvam_language,
            with_timestamps=True,
        )

    data = r if isinstance(r, dict) else r.model_dump()

    segments: list[TranscriptSegment] = []
    ts = data.get("timestamps") or {}
    starts = ts.get("start_time_seconds") or []
    ends = ts.get("end_time_seconds") or []
    texts = ts.get("words") or ts.get("sentences") or []

    for i, (txt, st, en) in enumerate(zip(texts, starts, ends)):
        txt = str(txt).strip()
        if txt:
            segments.append(
                TranscriptSegment(
                    index=len(segments),
                    start=round(float(st), 3),
                    end=round(float(en), 3),
                    text=txt,
                )
            )

    if not segments:
        # No timestamps came back — keep the text so the run still produces
        # something usable rather than failing outright.
        full = (data.get("transcript") or "").strip()
        if not full:
            raise RuntimeError(f"Sarvam returned nothing usable: {data}")
        segments = [TranscriptSegment(index=0, start=0.0, end=0.0, text=full)]
        print("      warning: Sarvam returned no timestamps; clip cutting "
              "will not work on this transcript")

    return Transcript(
        language=data.get("language_code") or config.sarvam_language,
        segments=segments,
        words=[],   # Sarvam does not provide word-level timings
    )