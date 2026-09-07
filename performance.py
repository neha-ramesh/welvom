#!/usr/bin/env python3
"""Step 7 CLI — pull our own posts' performance and store it.

    # weekly pull
    python performance.py --client davanam_jewellers

    # record that a clip went out as a particular post (do this at publish time)
    python performance.py --client davanam_jewellers --register 17912345678901234 \
        --clips data/output/51f11977cfc8_clips_G.json --rank 1

    # what the numbers say so far
    python performance.py --client davanam_jewellers --learnings

Run this weekly. Instagram will not tell you what a post's reach was last week
if you did not ask last week, so a week skipped is a week gone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.config import config
from pipeline.performance import (
    attach_sources,
    fetch_apify,
    fetch_official,
    learnings,
    register,
    save_snapshot,
)


def registry_path(client: str) -> Path:
    return Path(config.performance_dir) / f"{client}_published.json"


def do_register(args) -> int:
    if not args.clips or args.rank is None:
        print("--register needs --clips and --rank", file=sys.stderr)
        return 1
    clips = json.loads(Path(args.clips).read_text(encoding="utf-8"))
    clip = next((c for c in clips if c.get("rank") == args.rank), None)
    if not clip:
        print(f"No clip with rank {args.rank} in {args.clips}", file=sys.stderr)
        return 1
    clip["source_video"] = Path(args.clips).stem
    register(registry_path(args.client), args.register, clip, args.permalink)
    print(f"registered media {args.register} -> clip #{args.rank} "
          f"({clip.get('topic') or 'no topic'}, {clip.get('duration')}s)")
    return 0


def report(stats, matched: int) -> None:
    if stats.error:
        print(f"\nFAILED: {stats.error}", file=sys.stderr)
        return
    print(f"\n@{stats.username}  {stats.followers:,} followers  "
          f"{len(stats.posts)} posts  ({matched} from our pipeline)")

    has_insights = any(p.reach for p in stats.posts)
    if not has_insights:
        print("\n  No reach or save data. Either the backend is 'apify' (public")
        print("  data only) or the token is missing instagram_manage_insights.")

    print("\n" + "=" * 84)
    print(f"{'posted':<12}{'kind':<8}{'views':>9}{'reach':>9}{'likes':>8}"
          f"{'saves':>7}{'save%':>8}{'from our pipeline':>22}")
    print("-" * 84)
    for p in sorted(stats.posts, key=lambda x: x.posted_at, reverse=True):
        src = p.clip_source.get("topic", "") if p.clip_source else ""
        kind = "reel" if p.is_reel else p.media_type.lower()[:7]
        print(f"{p.posted_at[:10]:<12}{kind:<8}{p.views:>9,}{p.reach:>9,}"
              f"{p.likes:>8,}{p.saved:>7,}{p.save_rate() * 100:>7.2f}%"
              f"{src[:22]:>22}")

    ours = [p for p in stats.posts if p.clip_source and p.reach]
    if ours:
        print("\n" + "=" * 84)
        print("OUR CLIPS, BEST SAVE RATE FIRST")
        print("(saves per person reached — someone intending to come back is the")
        print(" strongest signal a clip was worth making)")
        print("=" * 84)
        for p in sorted(ours, key=lambda x: -x.save_rate())[:10]:
            s = p.clip_source
            print(f"\n  {p.save_rate() * 100:.2f}% save rate | {s.get('content_type', '?')}"
                  f" | {s.get('duration', '?')}s | reach {p.reach:,}")
            if p.is_reel and p.avg_watch_s():
                print(f"    avg watch {p.avg_watch_s():.1f}s of {s.get('duration', '?')}s")
            if s.get("hook"):
                print(f"    hook: {str(s['hook'])[:90]}")
            print(f"    {p.permalink}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", default="client")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--learnings", action="store_true",
                    help="what the data supports so far")
    ap.add_argument("--register", metavar="MEDIA_ID",
                    help="record that a clip was published as this post")
    ap.add_argument("--clips", type=Path, help="the *_clips_*.json it came from")
    ap.add_argument("--rank", type=int, help="which clip in that file")
    ap.add_argument("--permalink", default="")
    args = ap.parse_args()

    out_dir = Path(config.performance_dir)

    if args.register:
        return do_register(args)

    backend = config.performance_backend
    if backend == "apify":
        if not config.apify_token or not config.client_ig_username:
            print("Set APIFY_TOKEN and CLIENT_IG_USERNAME in .env", file=sys.stderr)
            return 1
        print(f"backend: apify (public data only — no reach, saves or watch time)")
        stats = fetch_apify(config.apify_token, config.client_ig_username,
                            limit=args.limit)
    else:
        if not (config.client_ig_user_id and config.client_ig_token):
            print("Set CLIENT_IG_USER_ID and CLIENT_IG_TOKEN in .env.\n"
                  "The client authorises your Meta app once; the token needs\n"
                  "instagram_manage_insights. Set PERFORMANCE_BACKEND=apify to\n"
                  "test the plumbing before that is arranged.", file=sys.stderr)
            return 1
        print("backend: meta graph api")
        stats = fetch_official(config.client_ig_user_id, config.client_ig_token,
                               limit=args.limit)

    matched = attach_sources(stats, registry_path(args.client))

    if args.learnings:
        l = learnings(stats)
        if not l["ready"]:
            print(f"\n{l['why']}")
            print("Keep publishing and keep registering. This is the number that")
            print("makes clip selection improve, and it only grows by waiting.")
            return 0
        print(f"\n{l['tracked']} clips measured. "
              f"Median save rate {l['median_save_rate'] * 100:.2f}%")
        print(f"\n{'content type':<20}{'n':>5}{'median save rate':>20}")
        for k, v in l["by_content_type"].items():
            print(f"{k:<20}{v['n']:>5}{v['median_save_rate'] * 100:>19.2f}%")
        return 0

    report(stats, matched)
    if not stats.error:
        path = save_snapshot(stats, out_dir, args.client)
        print(f"\nsnapshot saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())