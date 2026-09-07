#!/usr/bin/env python3
"""Turn a competitor snapshot into an HTML report you can send to the client.

    python report.py --client davanam_jewellers
    python report.py --client davanam_jewellers --open

Writes one self-contained .html file — no CDN, no server, no build step. It
opens in a browser, prints to PDF cleanly, and can be emailed as-is.

DESIGN NOTES

The palette comes from the subject rather than a template: these accounts keep
describing temple motifs set with red and green stones, so deep temple green
carries the structure, muted gold marks the leader, and kempu red flags the
accounts in trouble. Georgia throughout — one family, real tabular numerals for
a page made of percentages, and it renders the Kannada captions in this data
without falling back.

The hero is the festival ramp, not a headline number, because on the first real
dataset the ramp WAS the finding: a five-day escalation into Janmashtami that
produced 25,812 engagements in one day against 92 the week before.
"""

from __future__ import annotations

import argparse
import html
import math
import re
import statistics
import sys
import webbrowser
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

from pipeline.client_profile import ClientProfile, load_profile, profile_path
from pipeline.competitors import Competitor, growth, load_snapshots
from pipeline.config import config

IST = timedelta(hours=5, minutes=30)


def _day(iso: str, month_fmt: str = "%b") -> str:
    """'3 Sep' from an ISO date, without platform-specific format codes.

    strftime('%-d') is glibc only and raises on Windows; '%#d' is the Windows
    spelling. Building the string avoids having to care which one is running.
    """
    try:
        d = datetime.fromisoformat(iso[:10])
    except ValueError:
        return iso
    return f"{d.day} {d.strftime(month_fmt)}"

# A printed page has no affordance to expand anything, so every section is
# opened before the print dialog and restored afterwards. Kept out of the
# f-string because JS braces collide with format placeholders.
PRINT_JS = """
<script>
addEventListener("beforeprint", function () {
  document.querySelectorAll("details").forEach(function (d) {
    d.dataset.was = d.open ? "1" : "";
    d.open = true;
  });
});
addEventListener("afterprint", function () {
  document.querySelectorAll("details").forEach(function (d) {
    d.open = d.dataset.was === "1";
  });
});
</script>
"""

CSS = """
:root{
  --ink:#12261F;        /* deep temple green, the structural colour */
  --gold:#8A6B1F;       /* muted assay gold, not shiny */
  --kempu:#9E2B25;      /* temple red, reserved for trouble */
  --paper:#FAF8F3;
  --rule:#D8D2C4;
  --quiet:#5C6660;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font:16px/1.55 Georgia,'Iowan Old Style','Times New Roman',serif;
  font-variant-numeric:tabular-nums;
}
.wrap{max-width:60rem;margin:0 auto;padding:0 2rem 6rem}
header{background:var(--ink);color:var(--paper);padding:3.5rem 0 3rem;margin-bottom:3rem}
header .wrap{padding-bottom:0}
h1{font-size:2.1rem;font-weight:normal;letter-spacing:-.01em;margin:0 0 .4rem}
.sub{color:#B9C4BC;font-size:.95rem;margin:0}
h2{font-size:1.3rem;font-weight:normal;margin:3.5rem 0 .5rem;
   padding-bottom:.4rem;border-bottom:1px solid var(--rule)}
h2:first-of-type{margin-top:0}
.note{color:var(--quiet);font-size:.9rem;max-width:46rem;margin:.4rem 0 1.4rem}
table{width:100%;border-collapse:collapse;margin:1rem 0 .5rem;font-size:.95rem}
th{text-align:right;font-weight:normal;color:var(--quiet);font-size:.82rem;
   padding:0 0 .5rem;border-bottom:1px solid var(--rule)}
th:first-child{text-align:left}
td{padding:.6rem 0;border-bottom:1px solid rgba(216,210,196,.5);text-align:right}
td:first-child{text-align:left}
tr:last-child td{border-bottom:none}
.lead td{color:var(--gold)}
.weak td{color:var(--kempu)}
.bar{display:inline-block;height:.5rem;background:var(--ink);vertical-align:middle;
     margin-left:.5rem;min-width:1px}
.lead .bar{background:var(--gold)}
.weak .bar{background:var(--kempu)}
.ramp{display:flex;align-items:flex-end;gap:.7rem;height:11rem;margin:1.5rem 0 .5rem}
.ramp div{flex:1;display:flex;flex-direction:column;justify-content:flex-end;
          align-items:center;gap:.4rem}
.ramp .col{width:100%;background:var(--gold);min-height:2px}
.ramp .peak .col{background:var(--ink)}
.ramp .d{font-size:.75rem;color:var(--quiet);white-space:nowrap}
.ramp .v{font-size:.78rem}
.post{padding:1.2rem 0;border-bottom:1px solid rgba(216,210,196,.6)}
.post:last-child{border-bottom:none}
.post .m{font-size:.85rem;color:var(--quiet);margin-bottom:.35rem}
.post .mult{color:var(--gold)}
.post blockquote{margin:.5rem 0 0;padding-left:1rem;border-left:2px solid var(--rule);
                 color:#33443C;font-size:.92rem}
.post a{color:var(--ink);text-decoration:none;border-bottom:1px solid var(--rule)}
.hours{display:flex;gap:2px;margin:1rem 0 .3rem}
.hours div{flex:1;text-align:center;font-size:.7rem;color:var(--quiet)}
.hours .h{height:3.5rem;display:flex;align-items:flex-end;margin-bottom:.3rem}
.hours .h span{width:100%;background:var(--ink);opacity:.75;min-height:1px}
.hours .thin .h span{opacity:.18}
.hours .thin{color:#9AA39D}
.chart{width:100%;height:auto;margin:1.2rem 0 .5rem;overflow:visible}
.ax{font:11px Georgia,serif;fill:var(--quiet)}
.lbl{font:12px Georgia,serif;fill:var(--ink)}
.lbl.me{font-weight:bold}

.fgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(19rem,1fr));
       gap:1.6rem 2.4rem;margin:1.4rem 0 .5rem}
.facct h3{font-size:.95rem;font-weight:normal;margin:0 0 .5rem;color:var(--quiet)}
.fbar{display:flex;align-items:center;gap:.6rem;margin:.3rem 0;font-size:.8rem}
.fname{width:4.5rem;color:var(--quiet)}
.ftrack{flex:1;height:.75rem;background:rgba(216,210,196,.4)}
.ftrack span{display:block;height:100%}
.fval{width:3.4rem;text-align:right}

/* Evidence, folded. Available without being unavoidable. */
details{border-bottom:1px solid var(--rule)}
details[open]{padding-bottom:1.5rem}
summary{cursor:pointer;list-style:none;padding:1.1rem 0;font-size:1.15rem;
        display:flex;align-items:center;justify-content:space-between}
summary::-webkit-details-marker{display:none}
summary::after{content:"+";color:var(--quiet);font-size:1.3rem;line-height:1}
details[open] summary::after{content:"−"}
summary:hover{color:var(--gold)}
details h2{display:none}

ul{margin:.6rem 0 1.4rem;padding-left:1.15rem;max-width:46rem}
li{margin:.45rem 0;font-size:.95rem}
footer{margin-top:4rem;padding-top:1.2rem;border-top:1px solid var(--rule);
       color:var(--quiet);font-size:.85rem}
@media print{
  details{display:block !important}
  details > *:not(summary){display:revert !important}
  summary::after{display:none}
  header{background:none;color:var(--ink);border-bottom:2px solid var(--ink)}
  .sub{color:var(--quiet)} body{font-size:11pt}}
@media (max-width:40rem){.wrap{padding:0 1.2rem 4rem}h1{font-size:1.6rem}
  .ramp{height:8rem}}
"""


def esc(s: str) -> str:
    return html.escape(" ".join((s or "").split()))


def _scatter(comps: list[Competitor], client_handle: str = "") -> str:
    """Followers against engagement rate. One chart, no prose.

    An earlier version wrote sentences like "audience size is not what separates
    them", which is true and reads like a machine said it. Plotting the two
    axes makes the same point in a glance and lets the reader reach it
    themselves, which is the only way a client believes it.

    The x-axis is logarithmic because follower counts in a competitor set span
    an order of magnitude and a linear axis would crush everyone below the
    largest account into the left margin.
    """
    ok = [c for c in comps if not c.error and c.scorable and c.followers]
    if len(ok) < 3:
        return ""

    W, H = 760, 340
    L, R, T, B = 60, 30, 26, 56
    xs = [math.log10(c.followers) for c in ok]
    ys = [c.median_rate() * 100 for c in ok]
    x0, x1 = min(xs) - 0.15, max(xs) + 0.15
    y1 = max(ys) * 1.2 or 1

    def px(v: float) -> float:
        return L + (v - x0) / (x1 - x0) * (W - L - R)

    def py(v: float) -> float:
        return H - B - (v / y1) * (H - B - T)

    # gridlines
    grid = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        v = y1 * frac
        y = py(v)
        grid.append(f'<line x1="{L}" y1="{y:.0f}" x2="{W - R}" y2="{y:.0f}" '
                    f'stroke="var(--rule)" stroke-width="0.5"/>')
        grid.append(f'<text x="{L - 8}" y="{y + 4:.0f}" text-anchor="end" '
                    f'class="ax">{v:.1f}%</text>')

    ticks = []
    for p10 in range(int(x0), int(x1) + 2):
        if not (x0 <= p10 <= x1):
            continue
        x = px(p10)
        label = f"{10 ** p10 / 1000:.0f}k" if p10 >= 3 else f"{10 ** p10:.0f}"
        ticks.append(f'<text x="{x:.0f}" y="{H - B + 20}" text-anchor="middle" '
                     f'class="ax">{label}</text>')

    dots = []
    placed: list[tuple[float, float]] = []
    for c in sorted(ok, key=lambda c: -c.median_rate()):
        x, y = px(math.log10(c.followers)), py(c.median_rate() * 100)
        vid = c.format_mix().get("video", 0) + c.format_mix().get("carousel", 0)
        # Radius carries how much of the feed is moving content, so the reader
        # can see the big circles sit high without being told they correlate.
        r = 5 + vid * 13
        mine = c.username.lower() == client_handle.lower().lstrip("@")
        fill = "var(--ink)" if mine else "var(--gold)"

        # Labels flip to the left of the circle in the right half of the plot,
        # otherwise they run off the chart. And when two points sit close
        # together the second label is nudged below the first rather than
        # printed on top of it.
        right_half = x > (L + W - R) / 2
        lx = x - r - 7 if right_half else x + r + 7
        anchor = "end" if right_half else "start"
        ly = y + 4

        # Accounts near zero pile onto the axis, so their labels go above the
        # circle instead of beside it. The pile-up itself is the finding — these
        # accounts really are at the floor — but the text has to stay readable.
        near_floor = (H - B) - y < 26
        if near_floor:
            lx, anchor, ly = x, "middle", y - r - 6

        for _, prev_y in [(a, b) for a, b in placed if abs(a - x) < 160]:
            if abs(ly - prev_y) < 15:
                ly = prev_y - 16 if near_floor else prev_y + 16
        placed.append((x, ly))

        dots.append(
            f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r:.0f}" fill="{fill}" '
            f'opacity="{0.9 if mine else 0.6}"/>'
            f'<text x="{lx:.0f}" y="{ly:.0f}" text-anchor="{anchor}" '
            f'class="lbl{" me" if mine else ""}">{esc(c.username)}</text>'
        )

    return f"""
<h2>Followers do not buy attention</h2>
<p class="note">Each circle is an account. Across the bottom, audience size on a
log scale. Up the side, median engagement rate. Circle size is how much of the
feed is video or carousel rather than a still image.</p>
<svg viewBox="0 0 {W} {H}" class="chart" role="img"
     aria-label="Follower count against engagement rate for each competitor">
  {''.join(grid)}
  <line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="var(--ink)" stroke-width="1"/>
  {''.join(ticks)}
  <text x="{(L + W - R) / 2:.0f}" y="{H - 8}" text-anchor="middle" class="ax">followers</text>
  {''.join(dots)}
</svg>
"""


def _format_bars(comps: list[Competitor]) -> str:
    """Engagement by format, per account, as paired bars.

    The table version of this was correct and unreadable. Bars side by side let
    the eye do the comparison, and the pattern — video and carousel towering
    over image almost everywhere — arrives before any of the numbers do.
    """
    ok = [c for c in comps if not c.error and c.scorable and c.followers]
    if not ok:
        return ""
    ok.sort(key=lambda c: c.median_rate(), reverse=True)

    peak = 0.0
    for c in ok:
        peak = max([peak] + list(c.rate_by_format().values()))
    if not peak:
        return ""

    colours = {"video": "var(--ink)", "carousel": "var(--gold)",
               "image": "#C9C0AC"}
    rows = []
    for c in ok:
        r = c.rate_by_format()
        bars = []
        for fmt in ("video", "carousel", "image"):
            if fmt not in r:
                continue
            w = max(1, r[fmt] / peak * 100)
            bars.append(
                f'<div class="fbar"><span class="fname">{fmt}</span>'
                f'<span class="ftrack"><span style="width:{w:.1f}%;'
                f'background:{colours[fmt]}"></span></span>'
                f'<span class="fval">{r[fmt] * 100:.2f}%</span></div>'
            )
        if bars:
            rows.append(f'<div class="facct"><h3>@{esc(c.username)}</h3>'
                        f'{"".join(bars)}</div>')

    return f"""
<h2>Which format earns its place</h2>
<p class="note">Median engagement rate per format, per account, on one scale so
the accounts are comparable. Carousels are separated from single images because
they behave differently &mdash; the best post in this set is a carousel.</p>
<div class="fgrid">{''.join(rows)}</div>
"""


def _ramp(comps: list[Competitor]) -> str:
    """The leader's daily engagement, which is where campaign shape shows up."""
    ok = [c for c in comps if not c.error and c.scorable]
    if not ok:
        return ""
    top = max(ok, key=lambda c: c.median_rate())

    by_day: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    for p in top.scorable:
        if not p.timestamp:
            continue
        d = p.timestamp[:10]
        by_day[d] += p.engagement
        counts[d] += 1
    if len(by_day) < 3:
        return ""

    days = sorted(by_day)[-7:]
    peak = max(by_day[d] for d in days) or 1
    bars = []
    for d in days:
        v = by_day[d]
        h = max(2, int(v / peak * 150))
        cls = " peak" if v == peak else ""
        label = _day(d)
        bars.append(
            f'<div class="{cls.strip()}"><span class="v">{v:,}</span>'
            f'<span class="col" style="height:{h}px"></span>'
            f'<span class="d">{label}</span>'
            f'<span class="d">{counts[d]} post{"s" if counts[d] != 1 else ""}</span></div>'
        )

    peak_day = max(days, key=lambda d: by_day[d])
    peak_label = _day(peak_day, "%B")
    return f"""
<h2>How the leader builds to a festival</h2>
<p class="note">Daily engagement for @{esc(top.username)}, the strongest account
in this set. Read the shape rather than the totals: posting volume rises into
the date instead of landing on it, and {peak_label} carries
{counts[peak_day]} posts against one or two on the days before.</p>
<div class="ramp">{''.join(bars)}</div>
"""


def _leaderboard(comps: list[Competitor]) -> str:
    ok = [c for c in comps if not c.error and c.scorable]
    if not ok:
        return "<p class='note'>No accounts fetched successfully.</p>"
    ok.sort(key=lambda c: c.median_rate(), reverse=True)
    best = ok[0].median_rate() or 1

    rows = []
    for i, c in enumerate(ok):
        er = c.median_rate()
        mix = c.format_mix()
        cls = "lead" if i == 0 else ("weak" if er < best * 0.1 else "")
        w = max(1, int(er / best * 130))
        hidden = (f' <span class="d">({c.hidden_count()} hidden)</span>'
                  if c.hidden_count() else "")
        rows.append(
            f'<tr class="{cls}"><td>@{esc(c.username)}{hidden}</td>'
            f"<td>{c.followers:,}</td>"
            f'<td>{er * 100:.2f}%<span class="bar" style="width:{w}px"></span></td>'
            f"<td>{mix.get('video', 0) * 100:.0f}%</td>"
            f"<td>{mix.get('carousel', 0) * 100:.0f}%</td>"
            f"<td>{mix.get('image', 0) * 100:.0f}%</td></tr>"
        )
    return f"""
<h2>Who is actually being seen</h2>
<p class="note">Engagement rate is likes plus comments divided by followers.
Raw like counts cannot be compared across accounts of different sizes — 390
likes on 2,000 followers beats 1,100 likes on 186,000.</p>
<table><thead><tr><th>Account</th><th>Followers</th><th>Engagement</th>
<th>Video</th><th>Carousel</th><th>Image</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
"""


def _formats(comps: list[Competitor]) -> str:
    ok = [c for c in comps if not c.error and c.scorable]
    if not ok:
        return ""
    ok.sort(key=lambda c: c.median_rate(), reverse=True)
    rows = []
    for c in ok:
        r = c.rate_by_format()
        cell = lambda k: f"{r[k] * 100:.2f}%" if k in r else "&mdash;"
        rows.append(
            f"<tr><td>@{esc(c.username)}</td><td>{cell('video')}</td>"
            f"<td>{cell('carousel')}</td><td>{cell('image')}</td></tr>"
        )
    return f"""
<h2>Which format earns its place</h2>
<p class="note">Median engagement rate per format, per account. Carousels are
counted separately from images because they behave differently &mdash; on the
first run the single best post across five accounts was a carousel, not a reel.</p>
<table><thead><tr><th>Account</th><th>Video</th><th>Carousel</th>
<th>Image</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
"""


def _hours(comps: list[Competitor]) -> str:
    """When the outliers were posted, in IST."""
    hits: list[tuple[int, float]] = []
    total = 0
    for c in comps:
        if c.error or not c.scorable:
            continue
        for p in c.scorable:
            if not p.timestamp:
                continue
            total += 1
            try:
                t = datetime.fromisoformat(p.timestamp.replace("Z", "")) + IST
            except ValueError:
                continue
            hits.append((t.hour, p.rate(c.followers)))
    if len(hits) < 8:
        return ""

    bands = [(5, 9), (9, 12), (12, 15), (15, 19), (19, 24)]
    vals = []
    for a, b in bands:
        v = [er for h, er in hits if a <= h < b]
        vals.append((f"{a:02d}&ndash;{b:02d}", statistics.median(v) if v else 0, len(v)))
    peak = max(v for _, v, _ in vals) or 1

    # Bands with too few posts are drawn faint. Otherwise a band holding three
    # posts can produce the tallest bar on the page and the eye believes it long
    # before it reads the caveat underneath.
    cols = []
    for label, v, n in vals:
        h = max(1, int(v / peak * 56))
        thin_band = ' class="thin"' if n < 5 else ""
        cols.append(
            f'<div{thin_band}><div class="h"><span style="height:{h}px"></span></div>'
            f"{label} IST<br>{v * 100:.2f}%<br>n={n}</div>"
        )
    thin = [label for label, _, n in vals if n < 5]
    caveat = (
        f" Faded bars ({', '.join(thin)}) rest on fewer than five posts each, so "
        "they are a hint rather than a schedule. Two more weekly snapshots and "
        "they become evidence."
        if thin else ""
    )
    return f"""
<h2>When the good posts went out</h2>
<p class="note">Median engagement rate by time of day, Indian Standard Time,
across all {total} posts in this set.{caveat}</p>
<div class="hours">{''.join(cols)}</div>
"""


def _outliers(comps: list[Competitor], limit: int = 10) -> str:
    hits = []
    for c in comps:
        if c.error:
            continue
        med = c.median_rate()
        if not med:
            continue
        for p in c.outliers():
            hits.append((p.rate(c.followers) / med, c, p))
    if not hits:
        return ""
    hits.sort(key=lambda x: -x[0])

    items = []
    for mult, c, p in hits[:limit]:
        when = p.timestamp[:10]
        cap = esc(p.caption)[:230]
        items.append(f"""
<div class="post">
  <div class="m"><span class="mult">{mult:.1f}&times; their norm</span>
    &nbsp;@{esc(c.username)} &nbsp;{p.format} &nbsp;{max(0, p.likes):,} likes,
    {p.comments:,} comments &nbsp;{when}</div>
  <blockquote>{cap}{'&hellip;' if len(p.caption or '') > 230 else ''}</blockquote>
  <div class="m" style="margin-top:.5rem"><a href="{esc(p.permalink)}">View post</a></div>
</div>""")
    return f"""
<h2>Posts that beat their own account</h2>
<p class="note">Each post measured against its own account's median, not against
the other accounts. That answers what worked unusually well for them, which is
the version of the question that transfers to your feed.</p>
{''.join(items)}
"""


def _consistency(comps: list[Competitor]) -> str:
    """Gaps between posts.

    "No consistency" was the first pain point written by hand in both audits,
    and it is measurable from timestamps we already store — the median gap, and
    the longest silence. A brand posting every 8 days with a 3-week hole in the
    middle looks very different from one posting every 2 days.
    """
    rows = []
    for c in comps:
        if c.error or len(c.scorable) < 3:
            continue
        times = []
        for p in c.scorable:
            try:
                times.append(datetime.fromisoformat(p.timestamp.replace("Z", "")))
            except (ValueError, AttributeError):
                continue
        if len(times) < 3:
            continue
        times.sort()
        gaps = [(b - a).total_seconds() / 86400 for a, b in zip(times, times[1:])]
        span = (times[-1] - times[0]).days or 1
        rows.append((c.username, statistics.median(gaps), max(gaps),
                     len(times) / span * 7))
    if not rows:
        return ""

    rows.sort(key=lambda r: r[1])
    def gap(d: float) -> str:
        # "0.0 days" reads as broken data. Under a day it means several posts
        # the same day, which is a real and different behaviour worth naming.
        if d < 1:
            return f"{d * 24:.0f} hrs"
        return f"{d:.1f} days"

    body = "".join(
        f"<tr><td>@{esc(u)}</td><td>{gap(med)}</td><td>{gap(worst)}</td>"
        f"<td>{per_week:.1f}</td></tr>"
        for u, med, worst, per_week in rows
    )
    return f"""
<h2>How consistently they post</h2>
<p class="note">Days between posts, from the sample collected. The longest gap
matters as much as the median &mdash; a feed that goes quiet for three weeks
loses the reach it built, and the algorithm does not give it back for free.</p>
<table><thead><tr><th>Account</th><th>Typical gap</th>
<th>Longest gap</th><th>Posts/week</th></tr></thead><tbody>{body}</tbody></table>
"""


_TAG = re.compile(r"#(\w{2,40})")


def _hashtags(comps: list[Competitor], top: int = 18) -> str:
    """What the competition actually tags, and whether it correlates with reach.

    Both hand-written audits ended up prescribing a hashtag formula. This is the
    evidence for one: which tags appear in the posts that outperformed, rather
    than which tags are popular in general.
    """
    used: dict[str, list[float]] = {}
    tagless: list[float] = []
    for c in comps:
        if c.error or not c.followers:
            continue
        med = c.median_rate()
        for p in c.scorable:
            tags = {t.lower() for t in _TAG.findall(p.caption or "")}
            rel = (p.rate(c.followers) / med) if med else 0
            if not tags:
                tagless.append(rel)
                continue
            for t in tags:
                used.setdefault(t, []).append(rel)
    if not used:
        return ""

    ranked = sorted(
        ((t, len(v), statistics.median(v)) for t, v in used.items() if len(v) >= 2),
        key=lambda x: (-x[2], -x[1]),
    )[:top]
    if not ranked:
        return ""

    body = "".join(
        f"<tr><td>#{esc(t)}</td><td>{n}</td><td>{rel:.2f}&times;</td></tr>"
        for t, n, rel in ranked
    )
    extra = ""
    if len(tagless) >= 3:
        extra = (f" Posts with no hashtags at all ({len(tagless)} of them) ran at "
                 f"{statistics.median(tagless):.2f}&times; their account's norm.")
    return f"""
<h2>Hashtags that travel with a good post</h2>
<p class="note">Every tag used more than once, ranked by how the posts carrying
it performed against their own account's median. Above 1.00&times; means posts
with that tag beat the account's usual. This is correlation, not cause &mdash; a
festival tag looks strong because festival posts are strong.{extra}</p>
<table><thead><tr><th>Tag</th><th>Posts</th>
<th>vs account norm</th></tr></thead><tbody>{body}</tbody></table>
"""


def _profile_block(pr: ClientProfile) -> str:
    """Everything from the onboarding call. No data source can produce this."""
    if not pr:
        return ""
    out = []

    def ul(title: str, items: list[str], note: str = "") -> str:
        if not items:
            return ""
        lis = "".join(f"<li>{esc(i)}</li>" for i in items)
        n = f'<p class="note">{esc(note)}</p>' if note else ""
        return f"<h2>{title}</h2>{n}<ul>{lis}</ul>"

    if pr.revenue_note or pr.lead_routing or pr.moat:
        bits = [b for b in (pr.revenue_note, pr.lead_routing, pr.moat) if b]
        out.append("<h2>How the business actually makes money</h2>"
                   + "".join(f"<p class='note'>{esc(b)}</p>" for b in bits))

    out.append(ul("What is going wrong", pr.pain_points))
    out.append(ul("What they told us they want to post", pr.their_plans,
                  "Said on the onboarding call."))
    out.append(ul("Who we are writing for", pr.icp))

    if pr.goal or pr.cadence:
        out.append("<h2>The brief</h2>"
                   + (f"<p class='note'>{esc(pr.goal)}</p>" if pr.goal else "")
                   + (f"<p class='note'>{esc(pr.cadence)}</p>" if pr.cadence else ""))

    rules = list(pr.hard_rules)
    if pr.brand_colours:
        rules.append(f"Brand colours: {pr.brand_colours}")
    if pr.tagline:
        rules.append(f"Tagline: {pr.tagline}")
    out.append(ul("Hard rules", rules,
                  "From the onboarding form. These are not preferences."))

    if pr.reel_ideas:
        rows = "".join(
            f"<tr><td>{esc(r.idea)}</td><td>{esc(r.goal)}</td></tr>"
            for r in pr.reel_ideas
        )
        out.append(
            "<h2>What we would make</h2><p class='note'>Each format chosen for a "
            "job, not for variety.</p><table><thead><tr><th>Format</th>"
            f"<th>What it is for</th></tr></thead><tbody>{rows}</tbody></table>"
        )

    out.append(ul("What Welvom handles", pr.deliverables))

    tags = []
    for label, group in (("Local", pr.hashtags_local),
                         ("Service", pr.hashtags_service),
                         ("Broad", pr.hashtags_broad)):
        if group:
            tags.append(f"<tr><td>{label}</td><td>{esc(' '.join(group))}</td></tr>")
    if tags:
        out.append(
            "<h2>Hashtag set</h2><p class='note'>Local tags do the work for "
            "footfall; service tags catch intent; broad tags are for reach. Eight "
            "to fifteen per post, mixed across all three.</p>"
            f"<table><tbody>{''.join(tags)}</tbody></table>"
        )

    out.append(ul("Open questions", pr.open_questions,
                  "Unanswered after the call. Worth raising before the first post."))
    return "".join(x for x in out if x)


def _folded_profile(pr: ClientProfile) -> str:
    """Profile sections, each folded. Pain points stay open — that is the part a
    client reads first and argues with."""
    raw = _profile_block(pr)
    if not raw:
        return ""
    parts = re.split(r"(?=<h2>)", raw)
    out = []
    for part in parts:
        if not part.strip():
            continue
        m = re.search(r"<h2>(.*?)</h2>", part)
        label = m.group(1) if m else ""
        out.append(fold(label, part,
                        open_by_default=label.startswith("What is going wrong")))
    return "".join(out)


def _growth(out_dir: Path, client: str) -> str:
    g = growth(out_dir, client)
    if not g:
        return """
<h2>Follower growth</h2>
<p class="note">Needs a second snapshot. Instagram has no record of last week's
follower count, so growth only exists if it was stored &mdash; run this weekly
and the comparison appears here.</p>
"""
    rows = []
    for u, d in sorted(g.items(), key=lambda x: -x[1]["pct"]):
        cls = "lead" if d["pct"] > 0 else "weak"
        rows.append(
            f'<tr class="{cls}"><td>@{esc(u)}</td><td>{d["followers_was"]:,}</td>'
            f'<td>{d["followers_now"]:,}</td><td>{d["change"]:+,}</td>'
            f'<td>{d["pct"]:+.1f}%</td></tr>'
        )
    first = next(iter(g.values()))
    return f"""
<h2>Follower growth</h2>
<p class="note">Between {first['from']} and {first['to']}.</p>
<table><thead><tr><th>Account</th><th>Was</th><th>Now</th><th>Change</th>
<th>&nbsp;</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
"""


def fold(title: str, body: str, open_by_default: bool = False) -> str:
    """Wrap a section so the page is short by default.

    Folding is the right call here specifically because the reader already has
    the conclusion. These sections are the evidence for it, and evidence should
    be available rather than unavoidable.
    """
    if not body.strip():
        return ""
    # The section supplies its own <h2>; lift it into the summary line.
    m = re.search(r"<h2>(.*?)</h2>", body, re.DOTALL)
    label = m.group(1) if m else title
    inner = body.replace(m.group(0), "", 1) if m else body
    return (f'<details{" open" if open_by_default else ""}>'
            f"<summary>{label}</summary>{inner}</details>")


def build(comps: list[Competitor], client: str, out_dir: Path, stamp: str,
          profile: Optional[ClientProfile] = None) -> str:
    ok = sum(1 for c in comps if not c.error)
    posts = sum(len(c.scorable) for c in comps if not c.error)
    pretty = (profile.name if profile and profile.name
              else client.replace("_", " ").replace("-", " ").title())
    try:
        d = datetime.strptime(stamp, "%Y%m%d")
        shown = f"{d.day} {d.strftime('%B %Y')}"
    except ValueError:
        shown = stamp

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Competitor report &mdash; {esc(pretty)}</title>
<style>{CSS}</style>{PRINT_JS}</head><body>
<header><div class="wrap">
  <h1>What the competition is getting away with</h1>
  <p class="sub">{esc(pretty)} &nbsp;&middot;&nbsp; {shown} &nbsp;&middot;&nbsp;
     {ok} accounts, {posts} posts</p>
</div></header>
<div class="wrap">
{_scatter(comps, profile.handle if profile else "")}
{_format_bars(comps)}
{fold("The numbers behind it", _leaderboard(comps) + _formats(comps), open_by_default=False)}
{fold("The festival ramp", _ramp(comps), open_by_default=True)}
{fold("Timing", _hours(comps))}
{fold("Consistency", _consistency(comps))}
{fold("Hashtags", _hashtags(comps))}
{fold("Best posts", _outliers(comps))}
{fold("Growth", _growth(out_dir, client))}
{_folded_profile(profile)}
<footer>Public Instagram data, collected {shown}. Engagement rate is likes plus
comments over followers; posts with hidden like counts are excluded from every
figure rather than counted as zero.</footer>
</div></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", default="client")
    ap.add_argument("--open", action="store_true", help="open it in the browser")
    ap.add_argument("--init-profile", action="store_true",
                    help="write a filled-in profile template to edit")
    args = ap.parse_args()

    out_dir = Path(config.competitor_dir)
    snaps = load_snapshots(out_dir, args.client)
    if not snaps:
        print(f"No snapshots for '{args.client}' in {out_dir}.\n"
              f"Run: python competitors.py --client {args.client} <handles>",
              file=sys.stderr)
        return 1

    prof_path = profile_path(args.client, out_dir)
    if args.init_profile:
        from pipeline.client_profile import write_template
        print(f"wrote {write_template(prof_path)}\nEdit it, then re-run.")
        return 0

    profile = load_profile(prof_path)
    if not profile:
        print(f"No profile at {prof_path} — the report will cover the data only.")
        print(f"  python report.py --client {args.client} --init-profile")

    stamp, comps = snaps[-1]
    path = out_dir / f"{args.client}_report_{stamp}.html"
    path.write_text(build(comps, args.client, out_dir, stamp, profile),
                    encoding="utf-8")
    print(f"wrote {path}")
    if args.open:
        webbrowser.open(path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())