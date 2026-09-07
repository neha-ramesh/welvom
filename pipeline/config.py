"""Configuration. Everything switchable lives here so you can change backends
without touching pipeline code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # --- Transcription ---
    # "faster_whisper" (local, needs GPU for speed) | "groq" (hosted, no GPU)
    asr_backend: str = os.getenv("ASR_BACKEND", "faster_whisper")
    whisper_model: str = os.getenv("WHISPER_MODEL", "large-v3")
    # "cuda" | "cpu" | "auto"
    whisper_device: str = os.getenv("WHISPER_DEVICE", "auto")
    # float16 on GPU, int8 on CPU
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "whisper-large-v3-turbo")
    sarvam_api_key: str = os.getenv("SARVAM_API_KEY", "")
    sarvam_model: str = os.getenv("SARVAM_MODEL", "saaras:v3")
    sarvam_language: str = os.getenv("SARVAM_LANGUAGE", "en-IN")

    # Add sentence punctuation where the ASR left none. Cheap, and clip cutting
    # snaps to sentence boundaries, so without it early clips open mid-sentence.
    repunctuate: bool = os.getenv("REPUNCTUATE", "true").lower() == "true"

    # --- Which model does the thinking ---
    # "openai" = any OpenAI-compatible endpoint (incl. Gemini's compat layer)
    # "anthropic" = Claude's native API
    llm_backend: str = os.getenv("LLM_BACKEND", "openai")

    # Per-stage override. Segmentation is the one stage where model quality
    # showed a visible difference: Gemini produced only 4 segments for a
    # 6-minute video and one of them covered 48% of the runtime, which is what
    # forced --mode direct. Leave blank to use llm_backend.
    segment_backend: str = os.getenv("SEGMENT_BACKEND", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")

    # --- Semantic segmentation LLM ---
    # Any OpenAI-compatible endpoint. Defaults to Gemini's compat layer.
    llm_base_url: str = os.getenv(
        "LLM_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")

    # --- Vizard (edit mode: captions + vertical crop) ---
    vizard_api_key: str = os.getenv("VIZARDAI_API_KEY", "")
    vizard_template_id: str = os.getenv("VIZARD_TEMPLATE_ID", "")

    # --- Submagic (alternative editor: custom hook text, B-roll control) ---
    submagic_api_key: str = os.getenv("SUBMAGIC_API_KEY", "")
    submagic_template: str = os.getenv("SUBMAGIC_TEMPLATE", "")
    submagic_preset: str = os.getenv("SUBMAGIC_PRESET_ID", "")

    # --- Competitor tracking (step 1) ---
    # "official" = Instagram business_discovery (no block risk, needs app review,
    # cannot see personal accounts). "apify" = scraper fallback.
    competitor_backend: str = os.getenv("COMPETITOR_BACKEND", "official")
    ig_user_id: str = os.getenv("IG_USER_ID", "")
    ig_access_token: str = os.getenv("IG_ACCESS_TOKEN", "")
    apify_token: str = os.getenv("APIFY_TOKEN", "")
    competitor_dir: str = os.getenv("COMPETITOR_DIR", "data/competitors")

    # --- Performance tracking (step 7) ---
    # "official" = Meta Graph API: reach, saves, shares, watch time. Needs the
    # client to authorise the app. "apify" = public likes/comments only, which
    # is not enough to learn from — use it to test the plumbing.
    performance_backend: str = os.getenv("PERFORMANCE_BACKEND", "official")
    client_ig_user_id: str = os.getenv("CLIENT_IG_USER_ID", "")
    client_ig_token: str = os.getenv("CLIENT_IG_TOKEN", "")
    client_ig_username: str = os.getenv("CLIENT_IG_USERNAME", "")
    performance_dir: str = os.getenv("PERFORMANCE_DIR", "data/performance")

    # --- Storage: where cut clips live so Vizard can fetch them ---
    # "github" = free, predictable URLs, but PUBLIC and permanent. Fine for test
    # footage. "r2" = Cloudflare, no egress fees, deletable. Use before client
    # video is involved.
    storage_backend: str = os.getenv("STORAGE_BACKEND", "github")
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    github_repo: str = os.getenv("GITHUB_REPO", "")        # owner/name
    github_branch: str = os.getenv("GITHUB_BRANCH", "main")
    r2_account_id: str = os.getenv("R2_ACCOUNT_ID", "")
    r2_access_key: str = os.getenv("R2_ACCESS_KEY", "")
    r2_secret_key: str = os.getenv("R2_SECRET_KEY", "")
    r2_bucket: str = os.getenv("R2_BUCKET", "")
    r2_public_base: str = os.getenv("R2_PUBLIC_BASE", "")
    # Google Drive. A shared folder anyone can open, which is the point — a
    # GitHub raw URL is a file nobody outside the team knows what to do with.
    drive_folder_id: str = os.getenv("DRIVE_FOLDER_ID", "")
    # An OAuth client ID (Desktop app). A service-account key is detected and
    # accepted too, but only works with a Workspace Shared Drive — on a personal
    # account it fails with "Service Accounts do not have storage quota".
    drive_creds: str = os.getenv("DRIVE_CREDS", "google-oauth.json")
    drive_token: str = os.getenv("DRIVE_TOKEN", "drive-token.json")

    # --- Music bed (Vizard has no music parameter, so we mix it ourselves) ---
    music_dir: str = os.getenv("MUSIC_DIR", "assets/music")
    music_db: float = float(os.getenv("MUSIC_DB", "-18"))

    # --- Where finished clips are delivered ---
    # Point this at a Google Drive / OneDrive / Dropbox sync folder and the
    # finished clips appear in the client's shared folder with no API, no OAuth
    # and no upload code — the sync client already solved that problem.
    delivery_dir: str = os.getenv("DELIVERY_DIR", "data/ready")

    # --- Paths ---
    input_dir: str = os.getenv("INPUT_DIR", "data/input")
    output_dir: str = os.getenv("OUTPUT_DIR", "data/output")
    # Keep the extracted .wav around for debugging
    keep_audio: bool = os.getenv("KEEP_AUDIO", "false").lower() == "true"

    def resolved_compute_type(self) -> str:
        if self.whisper_compute_type:
            return self.whisper_compute_type
        return "float16" if self.resolved_device() == "cuda" else "int8"

    def resolved_device(self) -> str:
        """Resolve the device without importing torch.

        An earlier version checked `torch.cuda.is_available()`, but torch isn't a
        dependency here — faster-whisper uses CTranslate2. The import silently
        failed and everything fell back to CPU even on CUDA machines. Shelling
        out to nvidia-smi is ugly but honest: it tells you what's actually there.

        Set WHISPER_DEVICE explicitly to skip detection entirely.
        """
        if self.whisper_device != "auto":
            return self.whisper_device
        import shutil
        import subprocess

        if not shutil.which("nvidia-smi"):
            return "cpu"
        try:
            subprocess.run(
                ["nvidia-smi"], capture_output=True, timeout=5, check=True
            )
            return "cuda"
        except (subprocess.SubprocessError, OSError):
            return "cpu"


config = Config()