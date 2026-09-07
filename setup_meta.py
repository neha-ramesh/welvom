#!/usr/bin/env python3
"""Work out your Instagram account id and check the token actually works.

    python setup_meta.py EAAxxxxx...

Paste the token from the Graph API Explorer as the argument. This does the two
lookups by hand that are easy to get wrong, tells you which permissions are
missing, and prints the two lines to put in .env.

WHY THIS EXISTS

The Instagram id you need is not the username, not the Facebook Page id, and not
visible anywhere in the Instagram app. It only comes back from a Graph call
against the Page the account is linked to, which is two requests deep and
returns an unhelpful empty list when any prerequisite is missing. This names the
missing prerequisite instead.
"""

from __future__ import annotations

import sys

import requests

GRAPH = "https://graph.facebook.com/v23.0"

NEEDED = {
    "instagram_basic": "read the account and its posts",
    "instagram_manage_insights": "read reach, saves and watch time",
    "pages_show_list": "find the linked Facebook Page",
    "pages_read_engagement": "read through the Page",
}


def get(path: str, token: str, **params):
    params["access_token"] = token
    try:
        return requests.get(f"{GRAPH}/{path}", params=params, timeout=30).json()
    except requests.RequestException as e:
        return {"error": {"message": str(e)}}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        print("Get a token at developers.facebook.com/tools/explorer")
        print("  · select your app, top right")
        print("  · User Token")
        print("  · tick: " + ", ".join(NEEDED))
        return 1

    token = sys.argv[1].strip()

    # 1. Is the token real, and whose is it?
    me = get("me", token, fields="id,name")
    if "error" in me:
        e = me["error"]
        print(f"Token rejected: {e.get('message')}")
        if e.get("code") == 190:
            print("\nThat usually means it expired — Explorer tokens last about")
            print("an hour — or it was copied incompletely. Generate a new one.")
        return 1
    print(f"Token belongs to: {me.get('name')}\n")

    # 2. Which permissions did it actually get? The Explorer silently drops any
    #    the app has not been granted, so a token can look fine and still be
    #    unable to read insights.
    perms = get("me/permissions", token)
    granted = {p["permission"] for p in perms.get("data", [])
               if p.get("status") == "granted"}
    missing = [p for p in NEEDED if p not in granted]
    for p, why in NEEDED.items():
        print(f"  {'✓' if p in granted else '✗'} {p:28} {why}")
    if missing:
        print(f"\nMissing: {', '.join(missing)}")
        print("Regenerate the token in the Explorer with those ticked.")
        if "instagram_manage_insights" in missing:
            print("Without instagram_manage_insights you get likes and comments")
            print("and nothing else — which is no better than scraping.")
        print()

    # 3. Find the Page, then the Instagram account behind it.
    pages = get("me/accounts", token, fields="id,name,instagram_business_account")
    data = pages.get("data") or []
    if not data:
        print("No Facebook Pages on this account.\n")
        print("Instagram insights come through a Page, so you need one:")
        print("  1. facebook.com/pages/create — any name, nobody has to see it")
        print("  2. Instagram app → Settings → Account type and tools")
        print("     → Switch to professional account → Business or Creator")
        print("  3. Same menu → Sharing to other apps → Facebook → pick the Page")
        return 1

    found = False
    for pg in data:
        iga = pg.get("instagram_business_account")
        print(f"Page: {pg.get('name')}  ({pg.get('id')})")
        if not iga:
            print("   no Instagram account linked to this Page")
            continue
        found = True
        ig_id = iga["id"]
        info = get(ig_id, token, fields="username,followers_count,media_count")
        print(f"   Instagram: @{info.get('username')} · "
              f"{info.get('followers_count', 0):,} followers · "
              f"{info.get('media_count', 0)} posts")

        # 4. Prove insights actually come back, on a real post.
        media = get(ig_id, token, fields="media.limit(1){id,media_product_type}")
        items = (media.get("media") or {}).get("data") or []
        if not items:
            print("   no posts yet — publish one before tracking anything")
        else:
            mid = items[0]["id"]
            ins = get(f"{mid}/insights", token, metric="reach,saved,views")
            if "error" in ins:
                print(f"   insights refused: {ins['error'].get('message')}")
            else:
                vals = {r["name"]: r["values"][0].get("value")
                        for r in ins.get("data", [])}
                print(f"   insights on the latest post: {vals}")
                print("   working — that is the data step 7 records")

        print(f"\n   Put these in .env:")
        print(f"     CLIENT_IG_USER_ID={ig_id}")
        print(f"     CLIENT_IG_TOKEN={token[:12]}…  (the whole token)")
        print(f"     PERFORMANCE_BACKEND=official\n")

    if not found:
        print("\nA Page exists but no Instagram account is linked to it.")
        print("Instagram app → Settings → Account type and tools → Sharing to")
        print("other apps → Facebook → choose the Page.")
        return 1

    print("Explorer tokens expire in about an hour. Once this works, swap it")
    print("for a long-lived one (60 days):")
    print(f"  {GRAPH}/oauth/access_token?grant_type=fb_exchange_token"
          f"&client_id=APP_ID&client_secret=APP_SECRET&fb_exchange_token=TOKEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())