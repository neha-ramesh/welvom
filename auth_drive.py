#!/usr/bin/env python3
"""Authorise Drive once. Run on a machine with a browser.

    python auth_drive.py

Writes `drive-token.json`. After that every upload is silent — the token
refreshes itself and nothing ever asks you to log in again.

WHY THIS IS A SEPARATE SCRIPT

The consent step needs a browser. That is fine on your laptop and impossible on
a server, where `run_local_server` opens a browser that is not there and hangs
forever. So the approval happens here, once, and the resulting token file is
copied to wherever the app actually runs.

WHO IS LOGGING IN

You are, as the operator. The clips end up in your Drive and count against your
15 GB. Clients never authenticate — they open a folder link.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.config import config  # noqa: E402

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def main() -> int:
    creds_path = Path(config.drive_creds)
    token_path = Path(config.drive_token)

    if not creds_path.exists():
        print(f"No credentials at {creds_path.resolve()}\n")
        print("Get one from console.cloud.google.com:")
        print("  Credentials → Create credentials → OAuth client ID")
        print("  Application type: Desktop app")
        print(f"  Save the download as {creds_path}")
        return 1

    blob = json.loads(creds_path.read_text())
    if blob.get("type") == "service_account":
        print(f"{creds_path} is a service account key.\n")
        print("Service accounts own no storage, so uploads fail with")
        print('"Service Accounts do not have storage quota" unless you have a')
        print("Workspace Shared Drive. On a personal account you need an OAuth")
        print("client ID instead — Cloud Console → Credentials → OAuth client")
        print("ID → Desktop app.")
        return 1

    if token_path.exists():
        print(f"{token_path} already exists.")
        if input("Replace it? [y/N] ").strip().lower() != "y":
            return 0

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print("Run: pip install google-auth-oauthlib")
        return 1

    print("Opening a browser. Approve the request, then come back.\n")
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent",
                                  authorization_prompt_message="")
    token_path.write_text(creds.to_json())

    print(f"\nSaved {token_path.resolve()}")
    print("Uploads are silent from now on.")
    if config.drive_folder_id:
        print(f"Folder: https://drive.google.com/drive/folders/"
              f"{config.drive_folder_id}")
    else:
        print("Set DRIVE_FOLDER_ID in .env — the last part of the folder's URL.")
    print("\nDeploying somewhere else? Copy drive-token.json across with the "
          "app; that machine will not need a browser.")
    return 0


if __name__ == "__main__":
    sys.exit(main())