"""Step 1 — what are the competitors posting, and what is landing.

WHY THIS IS NOT A SCRAPER

Meta has an official endpoint for exactly this job: `business_discovery`. You
query any public Instagram Business or Creator account using your OWN access
token, and get their profile plus recent posts with like and comment counts. It
is the only official endpoint that supports competitor monitoring, and because
it is a real API there is no block risk — just a rate limit.

That removes the whole category of problem the user was worried about. No
rotating proxies, no browser automation, no account getting flagged.

WHAT IT WILL NEVER GIVE YOU

  - follower lists
  - reach, impressions or saves for accounts you do not own
  - anything at all from PERSONAL accounts

The last one is the real limitation in practice. A small local business — the
dental practice example — may well be on a personal account, and then it is
invisible here. `APIFY` backend exists for those cases.

THE METRIC THAT ACTUALLY MATTERS

Raw like counts are useless across accounts of different sizes. 500 likes on a
2,000-follower account is a hit; 500 likes on a 200,000-follower account is a
flop. So everything here is reported as ENGAGEMENT RATE — (likes + comments)
divided by followers. That is what makes a 5-competitor set comparable at all,
and it is why follower count has to be fetched alongside the posts rather than
treated as a separate lookup.

HISTORY IS SOMETHING YOU BUILD

The endpoint returns the CURRENT state only. There is no "followers 30 days
ago". If you want growth trends you store a snapshot every run and diff them
yourself, which is what `save_snapshot` is for.

SETUP, HONESTLY

business_discovery needs: an Instagram Business or Creator account, a linked
Facebook Page, a Meta app, and app review for public content access. Review
takes days to weeks. Until it clears, use `BACKEND=apify` to get moving.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from pydantic import BaseModel, Field

GRAPH = "https://graph.facebook.com/v22.0"

# Fields we ask for inside business_discovery. Keep this tight — every extra
# field is more payload against a 200/hour budget.
MEDIA_FIELDS = (
    "id,caption,like_count,comments_count,media_type,media_url,"
    "permalink,timestamp"
)
PROFILE_FIELDS = (
    "username,name,biography,website,profile_picture_url,"
    "followers_count,follows_count,media_count"
)


class Post(BaseModel):
    id: str
    caption: str = ""
    likes: int = 0
    comments: int = 0
    media_type: str = ""
    permalink: str = ""
    timestamp: str = ""

    @property
    def likes_hidden(self) -> bool:
        """Instagram lets an account hide its like count. Apify reports -1.

        These have to be excluded from any average, not counted as zero. On a
        real run two hidden posts dragged an account's median engagement down
        by 2.9x and moved it from second place to second-worst.
        """
        return self.likes < 0

    @property
    def engagement(self) -> int:
        return max(0, self.likes) + self.comments

    def rate(self, followers: int) -> float:
        """Engagement as a share of the audience. The only cross-account
        comparable number."""
        return (self.engagement / followers) if followers else 0.0

    @property
    def is_video(self) -> bool:
        return self.media_type.upper() in ("VIDEO", "REELS", "REEL")

    @property
    def format(self) -> str:
        """video / carousel / image.

        Carousels matter as their own category: on a real run the single
        best-performing post across five jewellery accounts was a carousel, not
        a reel, which a video/image split would have hidden.
        """
        t = self.media_type.upper()
        if t in ("VIDEO", "REELS", "REEL"):
            return "video"
        if t in ("SIDECAR", "CAROUSEL", "CAROUSEL_ALBUM"):
            return "carousel"
        return "image"


class Competitor(BaseModel):
    username: str
    name: str = ""
    biography: str = ""
    followers: int = 0
    follows: int = 0
    media_count: int = 0
    posts: list[Post] = Field(default_factory=list)
    fetched_at: str = ""
    error: str = ""

    # --- derived ---

    @property
    def scorable(self) -> list[Post]:
        """Posts we can actually measure — hidden like counts excluded."""
        return [p for p in self.posts if not p.likes_hidden]

    def hidden_count(self) -> int:
        return sum(1 for p in self.posts if p.likes_hidden)

    def median_rate(self) -> float:
        posts = self.scorable
        if not posts or not self.followers:
            return 0.0
        return statistics.median(p.rate(self.followers) for p in posts)

    def outliers(self, factor: float = 2.0) -> list[Post]:
        """Posts doing far better than this account's own norm.

        Comparing a post against its OWN account's median rather than against
        other accounts is the point. It answers "what worked unusually well for
        them", which is the only version of the question that transfers.
        """
        med = self.median_rate()
        if not med:
            return []
        return sorted(
            (p for p in self.scorable if p.rate(self.followers) >= med * factor),
            key=lambda p: p.rate(self.followers),
            reverse=True,
        )

    def format_mix(self) -> dict[str, float]:
        """Share of posts by format."""
        posts = self.scorable
        if not posts:
            return {}
        out: dict[str, float] = {}
        for p in posts:
            out[p.format] = out.get(p.format, 0.0) + 1
        return {k: v / len(posts) for k, v in out.items()}

    def rate_by_format(self) -> dict[str, float]:
        """Median engagement rate for each format this account uses.

        The question a client actually asks — 'should we make reels or
        carousels?' — is answered here, per account, in their own niche.
        """
        buckets: dict[str, list[float]] = {}
        for p in self.scorable:
            buckets.setdefault(p.format, []).append(p.rate(self.followers))
        return {k: statistics.median(v) for k, v in buckets.items() if v}

    def video_share(self) -> float:
        return self.format_mix().get("video", 0.0)


class CompetitorError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Official: Instagram business_discovery
# --------------------------------------------------------------------------- #


def fetch_official(
    ig_user_id: str,
    access_token: str,
    username: str,
    *,
    limit: int = 25,
    timeout: int = 30,
) -> Competitor:
    """One competitor, via Meta's own API.

    `ig_user_id` is YOUR Instagram Business account id; the token is yours too.
    The target only has to be a public Business or Creator account.
    """
    field = (
        f"business_discovery.username({username})"
        f"{{{PROFILE_FIELDS},media.limit({limit}){{{MEDIA_FIELDS}}}}}"
    )
    try:
        r = requests.get(
            f"{GRAPH}/{ig_user_id}",
            params={"fields": field, "access_token": access_token},
            timeout=timeout,
        )
        data = r.json()
    except requests.RequestException as e:
        return Competitor(username=username, error=f"request failed: {e}")

    if "error" in data:
        err = data["error"]
        msg = err.get("message", str(err))
        code = err.get("code")
        # 110 / "does not exist" almost always means a PERSONAL account, which
        # business_discovery cannot see at all. Worth naming, because the fix is
        # a different backend rather than a different query.
        if code == 110 or "does not exist" in msg.lower():
            msg += (
                " — this is usually a personal account (business_discovery only "
                "sees Business/Creator accounts). Try BACKEND=apify."
            )
        return Competitor(username=username, error=f"{code}: {msg}")

    bd = data.get("business_discovery")
    if not bd:
        return Competitor(username=username, error=f"no business_discovery in {data}")

    posts = [
        Post(
            id=str(m.get("id", "")),
            caption=m.get("caption", "") or "",
            likes=int(m.get("like_count") or 0),
            comments=int(m.get("comments_count") or 0),
            media_type=m.get("media_type", "") or "",
            permalink=m.get("permalink", "") or "",
            timestamp=m.get("timestamp", "") or "",
        )
        for m in (bd.get("media", {}).get("data") or [])
    ]

    return Competitor(
        username=bd.get("username", username),
        name=bd.get("name", "") or "",
        biography=bd.get("biography", "") or "",
        followers=int(bd.get("followers_count") or 0),
        follows=int(bd.get("follows_count") or 0),
        media_count=int(bd.get("media_count") or 0),
        posts=posts,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


# --------------------------------------------------------------------------- #
# Fallback: Apify, for personal accounts and pre-app-review testing
# --------------------------------------------------------------------------- #


APIFY_ACTOR = "apify/instagram-profile-scraper"


def _apify_post(m: dict[str, Any]) -> Post:
    """One post from an Apify item. Field names differ from Meta's."""
    return Post(
        id=str(m.get("id") or m.get("shortCode") or ""),
        caption=m.get("caption") or "",
        likes=int(m.get("likesCount") or 0),
        comments=int(m.get("commentsCount") or 0),
        media_type=(m.get("type") or "").upper(),
        permalink=m.get("url") or "",
        timestamp=m.get("timestamp") or "",
    )


def _apify_competitor(p: dict[str, Any], limit: int) -> Competitor:
    posts = [_apify_post(m) for m in (p.get("latestPosts") or [])[:limit]]
    return Competitor(
        username=p.get("username", "") or "",
        name=p.get("fullName", "") or "",
        biography=p.get("biography", "") or "",
        followers=int(p.get("followersCount") or 0),
        follows=int(p.get("followsCount") or 0),
        media_count=int(p.get("postsCount") or 0),
        posts=posts,
        fetched_at=datetime.now(timezone.utc).isoformat(),
    )


def fetch_apify_batch(
    apify_token: str,
    usernames: list[str],
    *,
    limit: int = 25,
    actor: str = APIFY_ACTOR,
) -> list[Competitor]:
    """All competitors in ONE actor run.

    Batching matters: Apify charges per run as well as per result, and a run has
    startup overhead of its own. Five separate runs for five handles is both
    slower and more expensive than one run with five handles in the input.

    Reads public data. Meta v. Bright Data (N.D. Cal. 2024) found scraping
    public logged-out data is not a CFAA violation, but it does breach
    Instagram's terms — a different risk category from business_discovery, not
    an equivalent one.
    """
    try:
        from apify_client import ApifyClient
    except ImportError as e:
        raise CompetitorError(
            "apify-client is not installed. Run: pip install apify-client"
        ) from e

    if not apify_token:
        raise CompetitorError("APIFY_TOKEN is not set")

    handles = [u.strip().lstrip("@") for u in usernames if u.strip()]
    if not handles:
        return []

    client = ApifyClient(apify_token)
    try:
        run = client.actor(actor).call(
            run_input={"usernames": handles, "resultsLimit": limit}
        )
    except Exception as e:
        raise CompetitorError(f"Apify run failed: {e}") from e

    if run is None:
        raise CompetitorError("Apify run returned nothing (it may have failed)")

    # The client has returned a dict historically and an object more recently.
    dataset_id = (
        run.get("defaultDatasetId")
        if isinstance(run, dict)
        else getattr(run, "default_dataset_id", None)
    )
    if not dataset_id:
        raise CompetitorError(f"No dataset id on the run: {run}")

    listed = client.dataset(dataset_id).list_items()
    items = listed.items if hasattr(listed, "items") else listed.get("items", [])

    by_name = {
        (it.get("username") or "").lower(): it for it in items if isinstance(it, dict)
    }

    out: list[Competitor] = []
    for h in handles:
        item = by_name.get(h.lower())
        if item is None:
            out.append(Competitor(
                username=h,
                error="not in the Apify results — private, renamed, or nonexistent",
            ))
        elif item.get("error"):
            out.append(Competitor(username=h, error=str(item["error"])))
        else:
            out.append(_apify_competitor(item, limit))
    return out


def fetch_apify(
    apify_token: str, username: str, *, limit: int = 25, actor: str = APIFY_ACTOR
) -> Competitor:
    """One account. Prefer fetch_apify_batch when you have several."""
    try:
        got = fetch_apify_batch(apify_token, [username], limit=limit, actor=actor)
    except CompetitorError as e:
        return Competitor(username=username, error=str(e))
    return got[0] if got else Competitor(username=username, error="no result")


# --------------------------------------------------------------------------- #
# Snapshots — the only way to get history
# --------------------------------------------------------------------------- #


def save_snapshot(
    competitors: list[Competitor], out_dir: Path, client: str = "client"
) -> Path:
    """Write one dated snapshot.

    The API has no concept of "followers last month", so growth has to be
    computed from stored snapshots. Run this weekly and the diff becomes
    available; skip a week and that week is gone permanently.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = out_dir / f"{client}_competitors_{stamp}.json"
    path.write_text(
        json.dumps([c.model_dump() for c in competitors], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load_snapshots(out_dir: Path, client: str = "client") -> list[tuple[str, list[Competitor]]]:
    """Every stored snapshot, oldest first, as (date, competitors)."""
    out: list[tuple[str, list[Competitor]]] = []
    for p in sorted(Path(out_dir).glob(f"{client}_competitors_*.json")):
        stamp = p.stem.rsplit("_", 1)[-1]
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
            out.append((stamp, [Competitor(**r) for r in rows]))
        except (ValueError, TypeError):
            continue
    return out


def growth(out_dir: Path, client: str = "client") -> dict[str, dict[str, Any]]:
    """Follower change between the oldest and newest snapshot."""
    snaps = load_snapshots(out_dir, client)
    if len(snaps) < 2:
        return {}
    (d0, first), (d1, last) = snaps[0], snaps[-1]
    old = {c.username: c.followers for c in first if c.followers}
    result: dict[str, dict[str, Any]] = {}
    for c in last:
        if c.username in old and old[c.username]:
            was = old[c.username]
            result[c.username] = {
                "from": d0,
                "to": d1,
                "followers_was": was,
                "followers_now": c.followers,
                "change": c.followers - was,
                "pct": round((c.followers - was) / was * 100, 2),
            }
    return result