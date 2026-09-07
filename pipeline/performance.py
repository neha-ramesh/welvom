"""Step 7 — what did our own clips actually do.

WHY THIS IS THE PIECE THAT COMPOUNDS

Step 1 tells you what competitors did. Anyone can buy that. This tells you what
YOUR clips did for THIS client, joined back to how each clip was made — its
topic, format, duration, hook, and which pipeline run produced it. Nobody else
can have that, and it is the only input step 8 has.

It also cannot be backfilled. Instagram will not tell you what a post's reach
was last week if you did not ask last week. Every week this does not run is a
week of evidence gone.

TWO BACKENDS, AND THEY ARE NOT EQUIVALENT

  official   Meta Graph API. Needs the client to authorise your app, but returns
             reach, saved, shares and watch time — the numbers that only the
             account owner can see.

  apify      Scrapes the public profile. Returns likes and comments and nothing
             else. This is the SAME data step 1 already collects about
             competitors, so it tells you almost nothing new about your own
             account. Useful to get the plumbing working before OAuth is set up;
             not useful as a destination.

METRIC NAMES CHANGED, AND THE OLD ONES NOW ERROR

Meta deprecated `impressions` and `plays` on 21 April 2025 and replaced both
with a single `views` metric that works across reels, carousels and stills.
Requesting `impressions` on media created after 2 July 2024 returns an error
rather than a number, so any code written against the old names fails outright
rather than degrading.

The distinction worth keeping straight: views counts every appearance including
replays and repeat viewers; reach counts unique accounts once each. Views is
always the larger number, and Meta describes reach as estimated.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests
from pydantic import BaseModel, Field

# Meta auto-upgrades deprecated versions and warns in a response header, so a
# pinned old version keeps working but silently runs on something else. Pin a
# current one and bump it deliberately.
GRAPH = "https://graph.facebook.com/v23.0"

MEDIA_FIELDS = "id,caption,media_type,media_product_type,permalink,timestamp,like_count,comments_count"

# Requested per media type, because asking for a metric a format does not
# support fails the WHOLE call rather than omitting that one field.
BASE_METRICS = ["reach", "saved", "shares", "total_interactions", "views"]
REEL_EXTRA = ["ig_reels_avg_watch_time", "ig_reels_video_view_total_time"]

# Never send these. They return an error, not an empty value.
DEAD_METRICS = {"impressions", "plays", "clips_replays_count",
                "ig_reels_aggregated_all_plays_count", "video_views"}


class PostStats(BaseModel):
    media_id: str
    permalink: str = ""
    caption: str = ""
    media_type: str = ""
    product_type: str = ""      # FEED / REELS / STORY
    posted_at: str = ""

    likes: int = 0
    comments: int = 0
    reach: int = 0
    views: int = 0
    saved: int = 0
    shares: int = 0
    total_interactions: int = 0
    avg_watch_time_ms: int = 0
    total_watch_time_ms: int = 0

    measured_at: str = ""
    insights_error: str = ""

    # Filled in from the publish registry when the post came from our pipeline.
    clip_source: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_reel(self) -> bool:
        return self.product_type.upper() in ("REELS", "CLIPS")

    def save_rate(self) -> float:
        """Saves per person reached. The strongest signal of genuine value —
        a save means someone intends to come back to it."""
        return (self.saved / self.reach) if self.reach else 0.0

    def share_rate(self) -> float:
        return (self.shares / self.reach) if self.reach else 0.0

    def engagement_rate(self, followers: int) -> float:
        base = self.total_interactions or (self.likes + self.comments + self.saved)
        return (base / followers) if followers else 0.0

    def avg_watch_s(self) -> float:
        return self.avg_watch_time_ms / 1000.0

    def hours_since(self) -> Optional[float]:
        if not self.posted_at:
            return None
        try:
            t = datetime.fromisoformat(self.posted_at.replace("Z", "+00:00"))
        except ValueError:
            return None
        return (datetime.now(timezone.utc) - t).total_seconds() / 3600.0


class AccountStats(BaseModel):
    username: str = ""
    followers: int = 0
    posts: list[PostStats] = Field(default_factory=list)
    fetched_at: str = ""
    error: str = ""


class PerformanceError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Official: Meta Graph API
# --------------------------------------------------------------------------- #


def _insights(media_id: str, token: str, metrics: list[str]) -> tuple[dict, str]:
    """Insights for one post. Returns (values, error).

    Falls back to a smaller metric set when the API rejects the request, which
    it does when a format does not support one of them — the call fails whole
    rather than skipping the unsupported field.
    """
    wanted = [m for m in metrics if m not in DEAD_METRICS]
    attempts = [wanted, BASE_METRICS, ["reach", "views"]]
    last_error = "insights unavailable for this media"

    for i, attempt in enumerate(attempts):
        try:
            r = requests.get(
                f"{GRAPH}/{media_id}/insights",
                params={"metric": ",".join(attempt), "access_token": token},
                timeout=30,
            )
            data = r.json()
        except requests.RequestException as e:
            return {}, f"request failed: {e}"

        if "error" not in data:
            out: dict[str, int] = {}
            for row in data.get("data", []):
                vals = row.get("values") or [{}]
                out[row.get("name", "")] = int(vals[0].get("value") or 0)
            return out, ""

        # Keep the message from the narrowest attempt — it is the most specific
        # about what is actually unavailable. An earlier version compared
        # `attempt is ["reach", "views"]`, which is an identity check against a
        # fresh list and therefore never true, so the real error was discarded
        # and every failure reported the same generic string.
        last_error = data["error"].get("message", last_error)
        if i == len(attempts) - 1:
            return {}, last_error

    return {}, last_error


def fetch_official(
    ig_user_id: str, token: str, *, limit: int = 25, with_insights: bool = True
) -> AccountStats:
    """Our own account's recent posts plus their insights."""
    try:
        r = requests.get(
            f"{GRAPH}/{ig_user_id}",
            params={
                "fields": f"username,followers_count,media.limit({limit}){{{MEDIA_FIELDS}}}",
                "access_token": token,
            },
            timeout=30,
        )
        data = r.json()
    except requests.RequestException as e:
        return AccountStats(error=f"request failed: {e}")

    if "error" in data:
        err = data["error"]
        hint = ""
        if err.get("code") in (100, 190):
            hint = (" Check the token has instagram_manage_insights and that the "
                    "account is a Business or Creator profile linked to a Page.")
        return AccountStats(error=f"{err.get('code')}: {err.get('message')}{hint}")

    now = datetime.now(timezone.utc).isoformat()
    posts: list[PostStats] = []

    for m in (data.get("media", {}).get("data") or []):
        p = PostStats(
            media_id=str(m.get("id", "")),
            permalink=m.get("permalink", "") or "",
            caption=m.get("caption", "") or "",
            media_type=m.get("media_type", "") or "",
            product_type=m.get("media_product_type", "") or "",
            posted_at=m.get("timestamp", "") or "",
            likes=int(m.get("like_count") or 0),
            comments=int(m.get("comments_count") or 0),
            measured_at=now,
        )
        if with_insights and p.media_id:
            metrics = BASE_METRICS + (REEL_EXTRA if p.is_reel else [])
            vals, err = _insights(p.media_id, token, metrics)
            p.reach = vals.get("reach", 0)
            p.views = vals.get("views", 0)
            p.saved = vals.get("saved", 0)
            p.shares = vals.get("shares", 0)
            p.total_interactions = vals.get("total_interactions", 0)
            p.avg_watch_time_ms = vals.get("ig_reels_avg_watch_time", 0)
            p.total_watch_time_ms = vals.get("ig_reels_video_view_total_time", 0)
            p.insights_error = err
        posts.append(p)

    return AccountStats(
        username=data.get("username", "") or "",
        followers=int(data.get("followers_count") or 0),
        posts=posts,
        fetched_at=now,
    )


# --------------------------------------------------------------------------- #
# Fallback: Apify (public data only)
# --------------------------------------------------------------------------- #


def fetch_apify(apify_token: str, username: str, *, limit: int = 25) -> AccountStats:
    """Public posts for our own account.

    Deliberately limited: this returns likes and comments and nothing else. No
    reach, no saves, no watch time. Those exist only through the authorised API,
    and they are the entire reason to run step 7 rather than just re-reading
    step 1's numbers. Use this to prove the plumbing, then move to `official`.
    """
    from .competitors import fetch_apify_batch

    got = fetch_apify_batch(apify_token, [username], limit=limit)
    if not got:
        return AccountStats(username=username, error="apify returned nothing")
    c = got[0]
    if c.error:
        return AccountStats(username=username, error=c.error)

    now = datetime.now(timezone.utc).isoformat()
    return AccountStats(
        username=c.username,
        followers=c.followers,
        posts=[
            PostStats(
                media_id=p.id,
                permalink=p.permalink,
                caption=p.caption,
                media_type=p.media_type,
                product_type="REELS" if p.is_video else "FEED",
                posted_at=p.timestamp,
                likes=max(0, p.likes),
                comments=p.comments,
                measured_at=now,
                insights_error="public data only — no reach, saves or watch time",
            )
            for p in c.posts
        ],
        fetched_at=now,
    )


# --------------------------------------------------------------------------- #
# The join: published post -> the clip that made it
# --------------------------------------------------------------------------- #


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    """Map of Instagram media id -> how that clip was made.

    Without this the numbers are just numbers. With it, a save rate can be
    attributed to a 42-second reel on a particular topic with a particular hook,
    which is what makes step 8 possible at all.
    """
    if not Path(path).exists():
        return {}
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    if isinstance(rows, dict):
        return rows
    return {str(r.get("media_id")): r for r in rows if r.get("media_id")}


def attach_sources(stats: AccountStats, registry_path: Path) -> int:
    """Attach clip metadata to every post we published. Returns how many matched."""
    reg = load_registry(registry_path)
    if not reg:
        return 0
    n = 0
    for p in stats.posts:
        src = reg.get(p.media_id)
        if src:
            p.clip_source = {
                k: v for k, v in src.items()
                if k in ("video_id", "rank", "topic", "content_type", "duration",
                         "hook", "score", "tag", "source_video")
            }
            n += 1
    return n


def register(
    registry_path: Path,
    media_id: str,
    clip: dict[str, Any],
    permalink: str = "",
) -> None:
    """Record that a clip was published as this Instagram post.

    Call this at publish time. It is one line, it costs nothing, and skipping it
    means the performance numbers can never be traced back to how the clip was
    made.
    """
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reg = load_registry(path)
    reg[str(media_id)] = {
        "media_id": str(media_id),
        "permalink": permalink,
        "published_at": datetime.now(timezone.utc).isoformat(),
        **{k: v for k, v in clip.items()
           if k in ("video_id", "rank", "topic", "content_type", "duration",
                    "hook", "score", "tag", "source_video")},
    }
    path.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Snapshots
# --------------------------------------------------------------------------- #


def save_snapshot(stats: AccountStats, out_dir: Path, client: str) -> Path:
    """One dated snapshot.

    Post metrics keep moving for days, so a clip measured at 24 hours and the
    same clip at 7 days are different data points, both worth having. Storing
    every pull rather than overwriting is what makes a 7-day curve possible.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = out_dir / f"{client}_performance_{stamp}.json"
    path.write_text(
        json.dumps(stats.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def load_snapshots(out_dir: Path, client: str) -> list[tuple[str, AccountStats]]:
    out: list[tuple[str, AccountStats]] = []
    for p in sorted(Path(out_dir).glob(f"{client}_performance_*.json")):
        try:
            out.append((p.stem.rsplit("_", 1)[-1],
                        AccountStats(**json.loads(p.read_text(encoding="utf-8")))))
        except (ValueError, TypeError):
            continue
    return out


def learnings(stats: AccountStats, min_posts: int = 6) -> dict[str, Any]:
    """What the numbers say so far, with an honest floor on sample size.

    Below `min_posts` this returns the reason it is staying quiet rather than a
    confident-looking breakdown of four posts. Three clips is not a finding.
    """
    tracked = [p for p in stats.posts if p.clip_source and p.reach]
    if len(tracked) < min_posts:
        return {
            "ready": False,
            "tracked": len(tracked),
            "need": min_posts,
            "why": (f"{len(tracked)} published clips have performance data. "
                    f"Patterns from fewer than {min_posts} are noise."),
        }

    by: dict[str, list[float]] = {}
    for p in tracked:
        key = p.clip_source.get("content_type") or p.clip_source.get("topic") or "?"
        by.setdefault(key, []).append(p.save_rate())

    import statistics
    return {
        "ready": True,
        "tracked": len(tracked),
        "median_save_rate": statistics.median(p.save_rate() for p in tracked),
        "by_content_type": {
            k: {"n": len(v), "median_save_rate": statistics.median(v)}
            for k, v in sorted(by.items()) if len(v) >= 2
        },
    }