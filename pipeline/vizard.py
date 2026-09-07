"""Step 4 — send our pre-cut clips to Vizard for captions and vertical crop.

Two things worth knowing before you use this.

**We use EDIT mode, not CLIP mode.** Vizard has two modes. Clip mode takes a long
video and picks moments itself — that's the part we're replacing. Edit mode
(`getClips: 0`) takes a short video and just makes it look good: vertical crop,
burned-in subtitles, an AI headline hook, optional B-roll and emoji. That's
exactly the division of labour we want, and it's cheaper because billing follows
minutes uploaded.

Hard limit: the submitted video must be **under 3 minutes**, or you get error
4005. Our clips are 20-75 seconds, so that's fine.

**Vizard needs a URL, not a file.** There is no file-upload endpoint. The clip
has to be reachable from the internet before you can submit it. The quickest
route for testing is a public Google Drive share link with `video_type=3` —
no storage account needed. See `README_STORAGE` at the bottom for the
production options.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import requests
from pydantic import BaseModel

BASE = "https://elb-api.vizard.ai/hvizard-server-front/open-api/v1/project"

# Vizard's own error code for "video too long for edit mode"
CODE_DONE = 2000        # finished
CODE_PROCESSING = 1000  # still working
ERR_TOO_LONG = 4005
MAX_SECONDS = 180


class VizardError(RuntimeError):
    pass


def submit_for_editing(
    api_key: str,
    video_url: str,
    *,
    lang: str = "auto",
    ext: str = "mp4",
    video_type: int = 1,       # 1 = direct .mp4 URL, 3 = Google Drive share link
    ratio: int = 1,            # 1 = 9:16 vertical
    subtitles: bool = True,
    headline: bool = True,     # the AI hook overlay — useful for clips that
                               # lack a natural opening line
    highlight_words: bool = True,
    emoji: bool = False,
    auto_broll: bool = False,
    remove_silence: bool = True,
    template_id: Optional[int] = None,
    project_name: Optional[str] = None,
    timeout: int = 60,
) -> str:
    """Submit one short clip. Returns the projectId to poll."""
    payload: dict[str, Any] = {
        "getClips": 0,          # EDIT mode, not clip mode
        "videoUrl": video_url,
        "videoType": video_type,
        "lang": lang,
        "ratioOfClip": ratio,
        "subtitleSwitch": 1 if subtitles else 0,
        "headlineSwitch": 1 if headline else 0,
        "highlightSwitch": 1 if highlight_words else 0,
        "emojiSwitch": 1 if emoji else 0,
        "autoBrollSwitch": 1 if auto_broll else 0,
        "removeSilenceSwitch": 1 if remove_silence else 0,
    }
    # ext is required only for videoType 1 (direct file URL)
    if video_type == 1:
        payload["ext"] = ext
    if template_id is not None:
        payload["templateId"] = template_id
    if project_name:
        payload["projectName"] = project_name

    r = requests.post(
        f"{BASE}/create",
        json=payload,
        headers={"VIZARDAI_API_KEY": api_key, "Content-Type": "application/json"},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()

    code = data.get("code")
    if code == ERR_TOO_LONG:
        raise VizardError(
            "Vizard rejected the clip as longer than 3 minutes (code 4005). "
            "Edit mode only accepts short videos."
        )
    # Field naming across Vizard responses isn't fully documented; check both.
    project_id = data.get("projectId") or data.get("data", {}).get("projectId")
    if not project_id:
        raise VizardError(f"No projectId in response: {data}")
    return str(project_id)


def poll(
    api_key: str,
    project_id: str,
    *,
    interval: float = 10.0,
    max_wait: float = 900.0,
    verbose: bool = True,
) -> dict:
    """Wait for the edit to finish. Returns the raw response.

    Vizard's status codes aren't published in the OpenAPI spec, so this treats
    "a video URL appeared in the response" as the completion signal rather than
    trusting a status field we can't verify. Print the response on your first
    run and tighten this once you've seen the real shape.
    """
    waited = 0.0
    while waited < max_wait:
        r = requests.get(
            f"{BASE}/query/{project_id}",
            headers={"VIZARDAI_API_KEY": api_key},
            timeout=60,
        )
        r.raise_for_status()
        data = r.json()

        code = data.get("code")
        clips = parse_result(data)
        if verbose:
            print(f"    [{waited:>4.0f}s] code={code} clips={len(clips)}")

        # 2000 = finished, 1000 = still processing, anything else = a problem.
        if code == CODE_DONE:
            if not clips:
                raise VizardError(f"Finished but returned no clips: {data}")
            return data
        if code == ERR_TOO_LONG:
            raise VizardError("Clip rejected as longer than 3 minutes (4005).")
        if code is not None and code != CODE_PROCESSING:
            raise VizardError(f"Vizard returned code {code}: {data}")

        time.sleep(interval)
        waited += interval

    raise VizardError(
        f"Timed out after {max_wait:.0f}s on project {project_id}. "
        f"Last response: {data}"
    )


class EditedClip(BaseModel):
    """One finished clip returned by Vizard."""

    video_url: str
    title: str = ""            # Vizard's AI headline — usable as a caption hook
    duration_s: float = 0.0
    video_id: Optional[int] = None
    editor_url: str = ""       # open in Vizard's UI to tweak by hand
    viral_score: Optional[str] = None
    viral_reason: str = ""
    topics: list[str] = []     # ready-made hashtag material
    transcript: str = ""


def parse_result(data: dict) -> list[EditedClip]:
    """Read the finished clips out of a query response.

    Response shape confirmed from a live run:
      {"code": 2000, "creditsUsed": 1, "projectId": ..., "videos": [{...}]}

    Note `videoUrl` is a signed CDN link with query parameters after the .mp4,
    so never test it with endswith(".mp4").
    """
    out: list[EditedClip] = []
    for v in data.get("videos") or []:
        url = v.get("videoUrl")
        if not url:
            continue
        topics = v.get("relatedTopic") or "[]"
        if isinstance(topics, str):
            try:
                topics = json.loads(topics)
            except (ValueError, TypeError):
                topics = []
        out.append(
            EditedClip(
                video_url=url,
                title=v.get("title") or "",
                duration_s=round((v.get("videoMsDuration") or 0) / 1000, 2),
                video_id=v.get("videoId"),
                editor_url=v.get("clipEditorUrl") or "",
                viral_score=str(v.get("viralScore")) if v.get("viralScore") else None,
                viral_reason=v.get("viralReason") or "",
                topics=[str(t) for t in topics],
                transcript=(v.get("transcript") or "").strip(),
            )
        )
    return out


def edit_clip(api_key: str, video_url: str, **kwargs) -> tuple[str, list[EditedClip]]:
    """Submit and wait. Returns (projectId, finished clips)."""
    pid = submit_for_editing(api_key, video_url, **kwargs)
    result = poll(api_key, pid)
    return pid, parse_result(result)


README_STORAGE = """
Vizard cannot accept a local file. Your clip must be at a public https URL that
Vizard's servers can download directly.

Options, roughly in order of how much sense they make:

  Cloudflare R2   No egress fees, S3-compatible, cheapest at volume. Best
                  long-term answer for an agency uploading clips constantly.
  AWS S3          Presigned URLs work fine. Watch egress costs.
  Google Drive    Works via videoType=3 with a public share link. Fine for
                  testing, fragile for production.

What will NOT work: a SharePoint link. Those require authentication, and Vizard
has no way to log in as your client.

So the real flow is: SharePoint (source) -> your machine -> ffmpeg cut ->
object storage -> Vizard. The storage step is unavoidable.
"""