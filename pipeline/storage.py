"""Put a cut clip somewhere Vizard can download it.

WHY THIS EXISTS

Vizard has no file-upload endpoint. It takes a URL and fetches the video itself,
so a clip sitting on a local disk is invisible to it. This is the one manual
step between a working pipeline and an unattended one.

TWO BACKENDS

  github   Free, and the URLs are predictable — raw.githubusercontent.com plus a
           path you construct. That predictability is the whole point: Google
           Drive gives every file a random opaque id, so nothing can build a URL
           and a human has to fetch each one by hand.

           Three limits worth knowing before a client video goes near it: a
           private repo has no raw URL Vizard can read, so everything uploaded
           is PUBLIC and indexable. Git keeps history, so deleting a clip does
           not reclaim the space and the repo only ever grows. And there is a
           100 MB per-file limit with a 5 GB soft cap on the repo.

  r2       Cloudflare R2. S3-compatible, and the reason to prefer it is egress:
           R2 charges none, while S3 bills every time Vizard downloads a clip.
           Free tier is 10 GB and a million writes a month, which at ~3 MB per
           clip is a few thousand clips. Deletes actually delete.

Use github to prove the loop this week with test footage. Move to r2 before real
client material is involved — the code is the same shape, so it is a config
change rather than a rewrite.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Optional

import requests

GITHUB_API = "https://api.github.com"


class StorageError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# GitHub
# --------------------------------------------------------------------------- #


def upload_github(
    local: Path,
    remote_path: str,
    *,
    token: str,
    repo: str,
    branch: str = "main",
    timeout: int = 180,
) -> str:
    """Commit a file and return its raw URL. `repo` is "owner/name"."""
    local = Path(local)
    if not local.exists():
        raise StorageError(f"{local} does not exist")

    size_mb = local.stat().st_size / 1_000_000
    if size_mb > 95:
        raise StorageError(
            f"{local.name} is {size_mb:.0f} MB; GitHub rejects files over 100 MB. "
            "Use STORAGE_BACKEND=r2."
        )

    url = f"{GITHUB_API}/repos/{repo}/contents/{remote_path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    # An existing file needs its blob sha to be replaced rather than rejected.
    sha: Optional[str] = None
    try:
        head = requests.get(url, params={"ref": branch}, headers=headers, timeout=30)
        if head.status_code == 200:
            sha = head.json().get("sha")
    except requests.RequestException:
        pass

    payload = {
        "message": f"clip: {remote_path}",
        "content": base64.b64encode(local.read_bytes()).decode(),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    r = requests.put(url, json=payload, headers=headers, timeout=timeout)
    if r.status_code not in (200, 201):
        try:
            detail = r.json().get("message", r.text[:300])
        except ValueError:
            detail = r.text[:300]
        hint = ""
        if r.status_code == 404:
            hint = (" Check the repo exists and the token has 'contents: write' "
                    "on it.")
        elif r.status_code == 409:
            hint = " Branch conflict — is the default branch 'main' or 'master'?"
        raise StorageError(f"GitHub {r.status_code}: {detail}{hint}")

    return f"https://raw.githubusercontent.com/{repo}/{branch}/{remote_path}"


# --------------------------------------------------------------------------- #
# Cloudflare R2
# --------------------------------------------------------------------------- #


def upload_r2(
    local: Path,
    remote_path: str,
    *,
    account_id: str,
    access_key: str,
    secret_key: str,
    bucket: str,
    public_base: str,
) -> str:
    """Upload to R2 and return the public URL.

    `public_base` is the r2.dev URL or your custom domain for the bucket. R2
    buckets are private by default, so without public access enabled the URL
    returns 403 and Vizard sees nothing — enable it in the bucket's settings
    before the first run.
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:
        raise StorageError("boto3 is not installed. Run: pip install boto3") from e

    local = Path(local)
    if not local.exists():
        raise StorageError(f"{local} does not exist")

    client = boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        # R2 ignores the region but the SDK insists on one.
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )

    ctype = mimetypes.guess_type(local.name)[0] or "video/mp4"
    try:
        client.upload_file(
            str(local), bucket, remote_path,
            ExtraArgs={"ContentType": ctype},
        )
    except Exception as e:
        raise StorageError(f"R2 upload failed: {e}") from e

    return f"{public_base.rstrip('/')}/{remote_path}"


# --------------------------------------------------------------------------- #
# Google Drive
# --------------------------------------------------------------------------- #


def _drive_service(creds_file: str, token_file: str):
    """Authenticated Drive client.

    TWO WAYS TO AUTHENTICATE, AND ONLY ONE WORKS ON A PERSONAL ACCOUNT

    A service account is the tidier option — a robot identity, no browser, no
    expiry — and it is what this originally used. It fails on a personal Gmail
    account with "Service Accounts do not have storage quota", because an
    uploaded file is owned by whoever uploaded it, and a service account owns no
    storage. Sharing the destination folder does not help: the folder's owner
    is not the file's owner.

    The documented fixes are a Shared Drive or domain-wide delegation, and both
    need Google Workspace. So on a personal account the only route is OAuth as
    yourself: the file is owned by you and counts against your 15 GB.

    That means one browser consent the first time. After that the refresh token
    in `token_file` keeps working indefinitely, so it stays unattended.

    A service-account key is still accepted — detected by the "type" field in
    the JSON — for anyone who does have a Shared Drive.
    """
    import json

    SCOPES = ["https://www.googleapis.com/auth/drive.file"]

    # Check the file before importing, so a missing key reports the missing key
    # rather than a module error that sends you installing packages you already
    # have.
    cf = Path(creds_file)
    if not cf.exists():
        raise StorageError(
            f"Google credentials not found at {cf.resolve()}. Download an OAuth "
            "client ID (Desktop app) from Cloud Console -> Credentials."
        )

    try:
        blob = json.loads(cf.read_text())
    except ValueError as e:
        raise StorageError(f"{cf} is not valid JSON") from e

    try:
        from googleapiclient.discovery import build
    except ImportError as e:
        raise StorageError(
            "Drive needs: pip install google-api-python-client google-auth "
            "google-auth-oauthlib"
        ) from e

    # A service-account key has "type": "service_account" at the top level.
    if blob.get("type") == "service_account":
        from google.oauth2 import service_account
        creds = service_account.Credentials.from_service_account_file(
            str(cf), scopes=SCOPES)
        return build("drive", "v3", credentials=creds, cache_discovery=False), True

    # Otherwise treat it as an OAuth client ID.
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as e:
        raise StorageError(
            "Drive needs: pip install google-api-python-client google-auth "
            "google-auth-oauthlib"
        ) from e

    tf = Path(token_file)
    creds = None
    if tf.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(tf), SCOPES)
        except ValueError:
            tf.unlink(missing_ok=True)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Opens a browser once. Everything after this is silent.
            flow = InstalledAppFlow.from_client_secrets_file(str(cf), SCOPES)
            creds = flow.run_local_server(port=0,
                                          prompt="consent",
                                          authorization_prompt_message="")
        tf.write_text(creds.to_json())

    return build("drive", "v3", credentials=creds, cache_discovery=False), False


def upload_drive(
    local: Path,
    name: str,
    *,
    folder_id: str,
    creds_file: str,
    token_file: str = "drive-token.json",
    make_public: bool = True,
) -> str:
    """Upload to a Drive folder and return a shareable link.

    SETUP, ONCE, ON A PERSONAL GOOGLE ACCOUNT

      1. console.cloud.google.com -> new project
      2. Enable the Google Drive API
      3. OAuth consent screen -> External -> add yourself as a test user
      4. Credentials -> Create credentials -> OAuth client ID -> Desktop app
      5. Download the JSON, save it as the DRIVE_CREDS path
      6. Folder id is the last part of drive.google.com/drive/folders/XXXX

    The first upload opens a browser to approve it. After that the saved token
    refreshes on its own.
    """
    try:
        from googleapiclient.http import MediaFileUpload
    except ImportError as e:
        raise StorageError(
            "Drive needs: pip install google-api-python-client google-auth "
            "google-auth-oauthlib"
        ) from e

    local = Path(local)
    if not local.exists():
        raise StorageError(f"{local} does not exist")

    svc, is_service_account = _drive_service(creds_file, token_file)
    media = MediaFileUpload(str(local), mimetype="video/mp4", resumable=True)

    try:
        f = svc.files().create(
            body={"name": name, "parents": [folder_id]},
            media_body=media,
            fields="id,webViewLink",
            supportsAllDrives=True,
        ).execute()
    except Exception as e:
        msg = str(e)
        if "storageQuota" in msg:
            msg = (
                "Service accounts own no storage, so an upload has nowhere to "
                "live. That only works with a Shared Drive, which needs Google "
                "Workspace. On a personal account, swap DRIVE_CREDS for an "
                "OAuth client ID (Cloud Console -> Credentials -> OAuth client "
                "ID -> Desktop app) and approve it once in the browser."
            )
        elif "notFound" in msg or "404" in msg:
            msg += ("\n  Check the folder id, and that this account can write "
                    "to that folder.")
        raise StorageError(msg) from e

    if make_public:
        try:
            svc.permissions().create(
                fileId=f["id"],
                body={"role": "reader", "type": "anyone"},
                supportsAllDrives=True,
            ).execute()
        except Exception:
            pass  # folder-level sharing may already cover it

    return f.get("webViewLink") or f"https://drive.google.com/file/d/{f['id']}/view"


def folder_link(folder_id: str) -> str:
    return f"https://drive.google.com/drive/folders/{folder_id}"


# --------------------------------------------------------------------------- #
# One interface
# --------------------------------------------------------------------------- #


def upload(local: Path, remote_path: str, config) -> str:
    """Upload via whichever backend is configured. Returns a public URL."""
    backend = getattr(config, "storage_backend", "github")

    if backend == "github":
        if not (config.github_token and config.github_repo):
            raise StorageError("Set GITHUB_TOKEN and GITHUB_REPO in .env")
        return upload_github(
            local, remote_path,
            token=config.github_token,
            repo=config.github_repo,
            branch=config.github_branch or "main",
        )

    if backend == "r2":
        missing = [k for k in ("r2_account_id", "r2_access_key", "r2_secret_key",
                               "r2_bucket", "r2_public_base")
                   if not getattr(config, k, "")]
        if missing:
            raise StorageError(
                "Missing in .env: " + ", ".join(m.upper() for m in missing)
            )
        return upload_r2(
            local, remote_path,
            account_id=config.r2_account_id,
            access_key=config.r2_access_key,
            secret_key=config.r2_secret_key,
            bucket=config.r2_bucket,
            public_base=config.r2_public_base,
        )

    if backend == "drive":
        # Reachable only if someone sets STORAGE_BACKEND=drive by hand. It is
        # the wrong tool for this job: this function exists to give Vizard a URL
        # it can download, and Drive returns a viewer page.
        raise StorageError(
            "STORAGE_BACKEND=drive will not work. This is the staging spot "
            "Vizard downloads the raw cut from, and it needs a direct file URL "
            "— use github or r2. Delivering finished clips to Drive is a "
            "separate setting, DRIVE_FOLDER_ID."
        )
    if backend == "_drive_unused":
        if not (config.drive_folder_id and config.drive_creds):
            raise StorageError("Set DRIVE_FOLDER_ID and DRIVE_CREDS in .env")
        return upload_drive(
            local, Path(remote_path).name,
            folder_id=config.drive_folder_id,
            creds_file=config.drive_creds,
        )

    raise StorageError(f"Unknown STORAGE_BACKEND: {backend!r}")


def upload_clips(
    clip_dir: Path, video_id: str, config, verbose: bool = True
) -> dict[int, str]:
    """Upload every clip_NN.mp4 in a folder. Returns {rank: public url}.

    Keyed by rank so the caller can match a URL back to the clip metadata
    without relying on filename parsing further down the line.
    """
    out: dict[int, str] = {}
    files = sorted(Path(clip_dir).glob("clip_*.mp4"))
    if not files:
        raise StorageError(f"No clip_*.mp4 files in {clip_dir}")

    for f in files:
        try:
            rank = int(f.stem.split("_")[-1])
        except ValueError:
            continue
        remote = f"clips/{video_id}/{f.name}"
        if verbose:
            print(f"  uploading {f.name} ({f.stat().st_size / 1e6:.1f} MB)...",
                  end="", flush=True)
        url = upload(f, remote, config)
        out[rank] = url
        if verbose:
            print(" done")
    return out