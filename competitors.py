#!/usr/bin/env python3
"""Step 1 CLI — pull competitor accounts and report what is landing.

    # one-off
    python competitors.py drsmithdental cityortho brightsmileclinic

    # a client's saved list, weekly
    python competitors.py --client acme-dental --from-file competitors.txt

    # follower growth between your stored snapshots
    python competitors.py --client acme-dental --growth

Engagement is reported as a RATE, not raw likes, because a set of competitors
with different follower counts is otherwise not comparable at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.competitors import (
    Competitor,
    CompetitorError,
    fetch_apify_batch,
    fetch_official,
    growth,
    save_snapshot,
)
from pipeline.config import config


def _line(c: Competitor) -> None:
    if c.error:
        print(f"  {c.username:<22} FAILED  {c.error[:100]}")
    else:
        print(f"  {c.username:<22} {c.followers:>10,} followers, "
              f"{len(c.posts)} posts")


def fetch_all(usernames: list[str], limit: int) -> list[Competitor]:
    backend = config.competitor_backend

    if backend == "apify":
        # One run for every handle — Apify charges per run as well as per
        # result, so looping would cost more and take longer.
        print("  starting one Apify run for all accounts (takes a minute)...")
        try:
            out = fetch_apify_batch(config.apify_token, usernames, limit=limit)
        except CompetitorError as e:
            print(f"  {e}", file=sys.stderr)
            return []
        for c in out:
            _line(c)
        return out

    out = []
    for u in usernames:
        u = u.strip().lstrip("@")
        if not u:
            continue
        c = fetch_official(config.ig_user_id, config.ig_access_token, u, limit=limit)
        _line(c)
        out.append(c)
    return out


def report(competitors: list[Competitor]) -> None:
    ok = [c for c in competitors if not c.error and c.posts]
    if not ok:
        print("\nNothing fetched successfully.")
        return

    print("\n" + "=" * 78)
    print(f"{'account':<24}{'followers':>11}{'median ER':>11}"
          f"{'video':>8}{'carousel':>10}{'image':>8}")
    print("-" * 78)
    for c in sorted(ok, key=lambda x: x.median_rate(), reverse=True):
        mix = c.format_mix()
        print(f"{c.username:<24}{c.followers:>11,}{c.median_rate() * 100:>10.2f}%"
              f"{mix.get('video', 0) * 100:>7.0f}%{mix.get('carousel', 0) * 100:>9.0f}%"
              f"{mix.get('image', 0) * 100:>7.0f}%")
        if c.hidden_count():
            print(f"{'':>24}  ({c.hidden_count()} post(s) with hidden like counts, "
                  f"excluded)")

    # Which format earns most, per account. This is the question clients ask.
    print("\n" + "=" * 78)
    print("MEDIAN ENGAGEMENT RATE BY FORMAT")
    print("=" * 78)
    print(f"{'account':<24}{'video':>12}{'carousel':>12}{'image':>12}")
    print("-" * 78)
    for c in sorted(ok, key=lambda x: x.median_rate(), reverse=True):
        r = c.rate_by_format()
        def cell(k):
            return f"{r[k] * 100:.2f}%" if k in r else "-"
        print(f"{c.username:<24}{cell('video'):>12}"
              f"{cell('carousel'):>12}{cell('image'):>12}")

    print("\n" + "=" * 72)
    print("POSTS THAT BEAT THEIR OWN ACCOUNT'S NORM BY 2x OR MORE")
    print("(compared against each account's own median, not against each other —")
    print(" that is what makes the answer transferable)")
    print("=" * 72)

    hits = []
    for c in ok:
        for p in c.outliers():
            hits.append((p.rate(c.followers) / (c.median_rate() or 1), c, p))
    if not hits:
        print("  none — every post is close to its account's norm")
        return

    for mult, c, p in sorted(hits, key=lambda x: x[0], reverse=True)[:15]:
        kind = p.format
        print(f"\n  {mult:.1f}x norm | @{c.username} | {kind} | "
              f"{p.likes:,} likes  {p.comments:,} comments")
        print(f"    {p.timestamp[:10]}  {p.permalink}")
        cap = " ".join((p.caption or "").split())
        if cap:
            print(f"    \"{cap[:150]}\"")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("usernames", nargs="*", help="competitor handles")
    ap.add_argument("--from-file", type=Path, help="one handle per line")
    ap.add_argument("--client", default="client", help="names the snapshot file")
    ap.add_argument("--limit", type=int, default=25, help="posts per account")
    ap.add_argument("--growth", action="store_true",
                    help="report follower change across stored snapshots")
    args = ap.parse_args()

    out_dir = Path(config.competitor_dir)

    if args.growth:
        g = growth(out_dir, args.client)
        if not g:
            print("Need at least two snapshots. Run without --growth first, "
                  "then again next week.")
            return 0
        print(f"{'account':<24}{'was':>12}{'now':>12}{'change':>12}{'%':>9}")
        print("-" * 69)
        for u, d in sorted(g.items(), key=lambda x: -x[1]["pct"]):
            print(f"{u:<24}{d['followers_was']:>12,}{d['followers_now']:>12,}"
                  f"{d['change']:>+12,}{d['pct']:>8.1f}%")
        return 0

    names = list(args.usernames)
    if args.from_file:
        names += [
            l.strip() for l in args.from_file.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")
        ]
    if not names:
        print("Give handles, or --from-file", file=sys.stderr)
        return 1

    backend = config.competitor_backend
    if backend == "apify" and not config.apify_token:
        print("APIFY_TOKEN is not set in .env", file=sys.stderr)
        return 1
    if backend == "official" and not (config.ig_user_id and config.ig_access_token):
        print("IG_USER_ID and IG_ACCESS_TOKEN are not set in .env.\n"
              "business_discovery needs an Instagram Business/Creator account, a\n"
              "linked Facebook Page, a Meta app, and app review for public content.\n"
              "Set COMPETITOR_BACKEND=apify to test before that clears.",
              file=sys.stderr)
        return 1

    print(f"backend: {backend}   accounts: {len(names)}")
    competitors = fetch_all(names, args.limit)
    report(competitors)

    path = save_snapshot(competitors, out_dir, args.client)
    print(f"\nsnapshot saved: {path}")
    print("Run again in a week and --growth will have something to compare.")
    return 0


if __name__ == "__main__":
    sys.exit(main())