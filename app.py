"""Welvom — a UI over the pipeline that already works.

    pip install streamlit
    streamlit run app.py

WHY STREAMLIT AND NOT VERCEL

Vercel runs serverless functions with a hard ceiling of 300 seconds. A 78-minute
video took 528 seconds end to end, there is no ffmpeg on serverless, and request
bodies cap at 4.5 MB against source files measured in gigabytes. Making it work
there means a job table, a queue, and a separate worker process — three moving
parts that exist only to work around the platform.

Streamlit is a normal long-running Python process. It imports the pipeline and
calls it. Nothing to queue, nothing to poll, ffmpeg already on the machine.

WHAT TO KNOW BEFORE EDITING THIS

Streamlit re-runs this entire file top to bottom on every interaction — every
click, every dropdown change. So anything expensive has to live in
`st.session_state`, which survives re-runs, or it will silently recompute. That
is the one concept that trips people up coming from ordinary scripts.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# Streamlit Cloud keeps secrets in st.secrets; everything in pipeline/ reads
# os.getenv, and config.py evaluates its defaults at import time. So the copy
# has to happen here, before that import, or every key reads as empty and the
# app silently reports nothing configured.
#
# Locally there are no secrets and .env is loaded by config.py as usual, so this
# is a no-op on your own machine.
import os

try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, (str, int, float, bool)):
            os.environ.setdefault(str(_k), str(_v))
except Exception:
    pass  # no secrets.toml — running locally

# Google's credential files are gitignored, correctly — they are secrets. Which
# means they do not exist on a deployed instance, and the Drive check reads as
# "not configured" while the folder id sits there looking perfectly fine.
#
# So the file CONTENTS travel as secrets and are written back to disk here.
# In Settings -> Secrets, paste each whole JSON as a TOML multi-line string
# using triple single quotes:
#
#   GOOGLE_OAUTH_JSON = <triple-quoted contents of google-oauth.json>
#   DRIVE_TOKEN_JSON  = <triple-quoted contents of drive-token.json>
#
# The token is the one that matters. It is what avoids a browser consent, and a
# browser consent is impossible on a server — so generate it locally with
# auth_drive.py first, then paste what that produced.
for _secret, _target in (
    ("GOOGLE_OAUTH_JSON", os.environ.get("DRIVE_CREDS", "google-oauth.json")),
    ("DRIVE_TOKEN_JSON", os.environ.get("DRIVE_TOKEN", "drive-token.json")),
):
    try:
        _blob = st.secrets.get(_secret)
    except Exception:
        _blob = None
    if _blob and not Path(_target).exists():
        try:
            Path(_target).write_text(str(_blob))
        except OSError:
            pass

st.set_page_config(page_title="Welvom", page_icon="◐", layout="wide")

# EVERY local import lives here, at module level, on purpose.
#
# Streamlit re-runs this file top to bottom on each interaction, and its file
# watcher tracks local packages between runs. Importing `pipeline.*` inside a
# button handler means the module is registered on one run and looked for on the
# next after the watcher has dropped it, which surfaces as `KeyError: 'pipeline'`
# and kills the app. Top-level imports are stable across reruns.
#
# This has to come after the secrets bridge above: config.py reads os.getenv when
# it is imported, so the environment must already be populated.
import report as _report
from clip import DECLICK_S, render
from pipeline.boundary import neighbours, refine
from pipeline.build import analyse_video, save
from pipeline.client_profile import load_profile, profile_path
from pipeline.competitors import (
    fetch_apify_batch,
    fetch_official as fetch_competitor,
    load_snapshots,
    save_snapshot as save_competitor_snapshot,
)
from pipeline.config import config
from pipeline.ffmpeg_setup import ensure_ffmpeg, which_report
from pipeline.models import VideoAnalysis
from pipeline.performance import (
    AccountStats,
    attach_sources,
    fetch_official as fetch_performance,
    register,
    save_snapshot as save_performance_snapshot,
)
from pipeline.probe import extract_metadata
from pipeline.select import select_clips
from pipeline.storage import (
    StorageError,
    folder_link,
    upload_clips,
    upload_drive,
)
from pipeline.vizard import parse_result, poll, submit_for_editing

# Put ffmpeg on PATH before any tab can shell out to it. Prefers a system
# install; falls back to the pip-installed static build, which is what makes
# this work on a host where apt is unavailable or broken.
_FFMPEG_OK = ensure_ffmpeg()

# --------------------------------------------------------------------------- #
# Styling
#
# Streamlit's defaults are recognisable enough that a demo looks like a
# Streamlit demo rather than a product. This pulls it toward the palette used in
# the client report — temple green and muted gold — so the tool and the
# deliverable look like one thing. It cannot be pushed all the way, and trying
# tends to break on the next Streamlit release.
# --------------------------------------------------------------------------- #

st.markdown("""
<style>
:root{--ink:#12261F;--gold:#8A6B1F;--kempu:#9E2B25;--paper:#FAF8F3;--rule:#D8D2C4}
.stApp{background:var(--paper)}
section[data-testid="stSidebar"]{background:#F2EFE7;border-right:1px solid var(--rule)}
h1,h2,h3{font-family:Georgia,serif !important;color:var(--ink) !important;
         font-weight:normal !important;letter-spacing:-.01em}
h1{font-size:2rem !important}
.stButton>button{background:var(--ink);color:var(--paper);border:0;border-radius:3px;
                 font-weight:500;padding:.5rem 1.4rem}
.stButton>button:hover{background:var(--gold);color:var(--paper)}
.stTabs [data-baseweb="tab"]{font-family:Georgia,serif;font-size:1rem}
.stTabs [aria-selected="true"]{color:var(--ink) !important}
[data-testid="stMetricValue"]{font-family:Georgia,serif;color:var(--ink)}
code{color:var(--gold)}

/* The timeline. Five slices carved out of a long recording — the one element
   that states what this product does without a caption. */
.tl{position:relative;height:34px;background:#EFEBE0;border:1px solid var(--rule);
    border-radius:2px;overflow:hidden;margin:.4rem 0}
.tl i{position:absolute;top:0;bottom:0;background:var(--gold);min-width:14px;
      display:flex;align-items:center;justify-content:center;font-style:normal;
      font-size:.7rem;color:var(--paper);font-weight:600;
      border-left:1px solid var(--paper);border-right:1px solid var(--paper)}
.ruler{display:flex;justify-content:space-between;color:#5C6660;font-size:.72rem;
       font-family:ui-monospace,Menlo,monospace}
.clip{border-left:2px solid var(--gold);padding:.1rem 0 .1rem .9rem;margin:.9rem 0}
.clip .tc{font-family:ui-monospace,Menlo,monospace;font-size:.78rem;color:#5C6660}
.clip .hook{font-size:.95rem;color:var(--ink);margin-top:.15rem}
</style>
""", unsafe_allow_html=True)


def tc(sec: float) -> str:
    h, m, s = int(sec // 3600), int(sec % 3600 // 60), int(sec % 60)
    return (f"{h:02d}:" if h else "") + f"{m:02d}:{s:02d}"


def timeline(clips, duration: float) -> str:
    """The source video with the chosen clips marked.

    A 43-second clip inside 78 minutes is 0.9% of the bar, so segments get a
    minimum width — otherwise they are two pixels wide and honest but invisible.
    """
    segs = "".join(
        f'<i style="left:{c["start"] / duration * 100:.2f}%;'
        f'width:{max(0.8, (c["end"] - c["start"]) / duration * 100):.2f}%">'
        f'{c["rank"]}</i>'
        for c in clips
    )
    return (f'<div class="tl">{segs}</div>'
            f'<div class="ruler"><span>00:00</span>'
            f'<span>{tc(duration / 2)}</span><span>{tc(duration)}</span></div>')


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("### welvom")
    st.caption("Long video in, reels out. Competitor audit on the side.")
    st.divider()


    # Which services are wired up is a deployment detail. A client seeing
    # "Groq (speech)" unticked learns nothing and worries anyway, so the whole
    # checklist lives behind this.
    internal = st.toggle(
        "Internal mode", value=False,
        help="Shows configuration and bookkeeping that clients do not need.")

    if internal:

        st.caption("**Wired up**")
        for label, ok in [
            ("Groq (speech)", bool(config.groq_api_key)),
            ("Gemini", bool(config.llm_api_key)),
            ("Claude", bool(config.anthropic_api_key)),
            ("Vizard", bool(config.vizard_api_key)),
            ("GitHub storage", bool(config.github_token and config.github_repo)),
            ("Apify", bool(config.apify_token)),
            ("Instagram insights", bool(config.client_ig_token
                                        and config.client_ig_user_id)),
        ]:
            st.write(f"{'✓' if ok else '·'} {label}")

        # Drive gets its own lines: there are three separate ways for it to be
        # not-quite-configured and one tick would hide which you have hit.
        _creds = Path(config.drive_creds)
        if not config.drive_folder_id:
            st.write("· Google Drive — no DRIVE_FOLDER_ID")
        elif not _creds.exists():
            st.write("· Google Drive — key file missing")
            st.caption(f"Looked for `{_creds}` in {_P.cwd()}")
        elif not Path(config.drive_token).exists():
            st.write("· Google Drive — not authorised")
            st.caption("Run `python auth_drive.py` once, locally.")
        else:
            st.write("✓ Google Drive")

        if st.button("Test the Drive connection", width="stretch"):
            try:
                probe = Path(tempfile.gettempdir()) / "welvom-test.txt"
                probe.write_text("welvom connection test")
                st.success(upload_drive(probe, "welvom-test.txt",
                                        folder_id=config.drive_folder_id,
                                        creds_file=config.drive_creds,
                                        token_file=config.drive_token))
            except Exception as e:
                st.error(str(e)[:400])

        # On Streamlit Cloud a missing secret looks identical to a typo in the
        # key name, so show what actually arrived.
        try:
            keys = sorted(st.secrets.keys())
            st.caption(f"{len(keys)} secret(s) loaded"
                       + (f": {', '.join(keys[:6])}…" if keys else ""))
        except Exception:
            st.caption("No secrets file — reading .env")
        st.caption(which_report())

tab_clip, tab_audit, tab_track = st.tabs(
    ["Clip a video", "Audit a client", "Track what worked"])

# --------------------------------------------------------------------------- #
# Clip
# --------------------------------------------------------------------------- #

with tab_clip:
    st.markdown("# Find the five minutes worth posting")
    st.caption("Drop in a recording of any length. It transcribes, picks the "
               "moments that stand alone, cuts on the breath, and sends them "
               "for captions.")

    # ffprobe and ffmpeg are shelled out to, so their absence surfaces as a
    # FileNotFoundError from subprocess with no hint about what is missing. On
    # Streamlit Cloud they come from packages.txt, which fails to install
    # whenever Debian's mirror has a stale release file — so this is worth
    # checking rather than letting someone hit Start and read a traceback.
    if not _FFMPEG_OK:
        st.error("Video tools are unavailable on this machine.")
        st.caption("Add `static-ffmpeg` to requirements.txt, or install ffmpeg "
                   "system-wide.")
        st.stop()

    # Clipping is the heavy tab: it holds the upload, extracts audio,
    # transcribes, and renders several files. Streamlit Community Cloud gives
    # 1 GB and wipes the disk on restart, so a real client recording will be
    # killed mid-run. Better to say so than to let it die at 80%.
    try:
        _total_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        _total_gb = 0.0
    if 0 < _total_gb < 2.5:
        st.warning(
            f"This machine has about {_total_gb:.1f} GB of memory. Clipping a "
            f"long recording needs more than that and will be killed part-way "
            f"through. Run this tab locally; the other two are fine here."
        )
        st.caption("A small VPS (about €4/month) runs all three without the "
                   "ceiling, if it needs to be hosted.")

    up = st.file_uploader("Video", type=["mp4", "mov", "m4v", "mkv"],
                          label_visibility="collapsed",
                          help="Up to 5 GB — the limit is raised in "
                               ".streamlit/config.toml")
    video_path = None
    if up:
        # Streamlit holds the whole upload in memory. Writing it with
        # `write_bytes(up.getbuffer())` briefly holds a second copy, so a 500 MB
        # file peaks near a gigabyte — which is the entire budget on Streamlit
        # Community Cloud. Copying in chunks keeps the peak flat, and dropping
        # the reference afterwards lets the buffer go.
        import gc
        import shutil as _sh

        tmp_dir = Path(tempfile.mkdtemp(prefix="welvom_"))
        video_path = tmp_dir / up.name
        with video_path.open("wb") as _out:
            _sh.copyfileobj(up, _out, length=1 << 22)
        up.seek(0)
        gc.collect()

    # No model picker. Which model runs at which stage was settled by testing —
    # Claude for choosing clips and their edges, Gemini's free tier for topic
    # grouping and punctuation repair — and re-opening it as a dropdown invites
    # someone to undo that by accident.
    c1, c2, c3 = st.columns([2, 1, 2])
    ctype = c1.selectbox("What kind of video is this?",
                         ["interview", "testimonial", "founder_talk", "generic"],
                         format_func=lambda k: {
                             "interview": "Interview",
                             "testimonial": "Customer testimonial",
                             "founder_talk": "Founder or expert talking",
                             "generic": "Something else",
                         }[k],
                         help="Changes what counts as a good clip. A testimonial "
                              "needs a problem and a specific result; a founder "
                              "talk needs a claim you could argue with.")
    # More candidates than anyone will post. A week is 10-14 slots at two a day,
    # and having spares is what makes "these three are weak" survivable without
    # re-running the whole video.
    n = c2.selectbox("How many clips?", [3, 5, 8, 12], index=2)
    # No half-run option. A "just cut it" mode sounds useful and is not: the
    # raw cuts are horizontal with no captions, so nobody can judge them as
    # reels, and anyone who likes them has to pay for the second half anyway.
    # One button, finished clips, and the choosing happens afterwards from the
    # folder — which is where it happens already, when scheduling.
    _creds_path = Path(config.drive_creds)
    drive_on = bool(config.drive_folder_id and _creds_path.exists())
    # Optional. On a hosted deployment there is nowhere useful to keep a copy
    # anyway — the disk is wiped on restart — so this defaults to off there.
    keep_local = c3.checkbox("Also keep a local copy",
                             value=not bool(config.drive_folder_id))
    dest = (c3.text_input("Folder", value=config.delivery_dir,
                          label_visibility="collapsed")
            if keep_local else tempfile.gettempdir())
    if drive_on:
        c3.caption(f"Delivering to [the Drive folder]"
                   f"({folder_link(config.drive_folder_id)})")
    elif not config.drive_folder_id:
        c3.caption("No DRIVE_FOLDER_ID in .env — clips stay local.")
    else:
        # The commonest cause on a deployed instance: the credential files are
        # gitignored, so they never got there. Naming both the path and the fix
        # beats a caption saying "not configured".
        c3.warning("Drive credentials missing — clips stay local.")
        c3.caption(f"Looked for `{_creds_path}` in `{Path.cwd()}`. Deployed? "
                   f"Add `GOOGLE_OAUTH_JSON` and `DRIVE_TOKEN_JSON` to Secrets "
                   f"with the file contents.")
    stop = "edit"

    if video_path:
        gb = video_path.stat().st_size / 1e9
        st.caption(f"{video_path.name} — "
                   + (f"{gb:.2f} GB" if gb >= 1 else f"{gb * 1000:.0f} MB"))

    # Two storage roles that are easy to mix up, and mixing them up fails
    # quietly. STORAGE_BACKEND is where Vizard FETCHES the raw cut from, so it
    # has to be a direct file URL — GitHub raw or R2. Drive's share link is a
    # viewer page, not a file, so Vizard downloads HTML and the run dies before
    # anything is ever delivered.
    if config.storage_backend == "drive":
        st.error(
            "STORAGE_BACKEND is set to `drive`, which cannot work: Vizard has "
            "to download the raw cut, and a Drive share link opens a viewer "
            "page rather than a file.\n\n"
            "Set `STORAGE_BACKEND=github` (or `r2`) — that is only a staging "
            "spot Vizard reads from. Delivery to Drive is separate and uses "
            "`DRIVE_FOLDER_ID`."
        )
        st.stop()

    if st.button("Start", type="primary", disabled=video_path is None):
        tmp = video_path


        # Settled by testing: Claude picks the clips and their exact edges,
        # Gemini handles the cheap stages inside run.py.
        config.llm_backend = "anthropic"
        out_dir = Path(config.output_dir)

        with st.status("Working…", expanded=True) as status:
            meta = extract_metadata(tmp)
            st.write(f"{meta.duration / 60:.0f} min · {meta.width}×{meta.height}")

            # The video id is a hash of the file, so a re-upload of the same
            # video skips the slowest and only paid stage.
            analysis_path = out_dir / f"{meta.video_id}.json"
            if analysis_path.exists():
                analysis = VideoAnalysis(
                    **json.loads(analysis_path.read_text(encoding="utf-8")))
                st.write(f"Transcript already on disk — reusing "
                         f"{len(analysis.transcript.segments)} chunks")
            else:
                st.write("Transcribing…")
                analysis = analyse_video(tmp)
                save(analysis, out_dir)
                st.write(f"{len(analysis.transcript.words):,} words")

            st.write("Reading the transcript and picking moments…")
            clips = select_clips(analysis, content_type=ctype, top_n=n,
                                 mode="direct")
            if not clips:
                status.update(label="No clips survived the filters", state="error")
                st.stop()

            st.write("Listening for the pause to cut on…")
            words = analysis.transcript.words
            for c in clips:
                lo, hi = neighbours(words, c.start, "start")
                c.start, _ = refine(tmp, c.start, lower=lo, upper=hi)
                lo, hi = neighbours(words, c.end, "end")
                c.end, _ = refine(tmp, c.end, lower=lo, upper=hi)
                c.duration = round(c.end - c.start, 2)

            clip_dir = out_dir / f"{meta.video_id}_clips_{ctype}"
            for c in clips:
                render(tmp, c, clip_dir)
            st.write(f"Cut {len(clips)} clips")

            urls = {}
            if stop in ("publish", "edit"):
                try:
                    st.write("Putting the clips where Vizard can reach them…")
                    urls = upload_clips(clip_dir, meta.video_id, config,
                                        verbose=False)
                    for c in clips:
                        c.url = urls.get(c.rank, "")
                except StorageError as e:
                    st.error(f"Could not stage the clips for Vizard: {e}")
                    st.caption("Vizard downloads each clip from a public URL, so "
                               "this step has to succeed before captions can be "
                               "added. The raw cuts are still on disk.")

            finished = []
            ready_dir = Path(dest) / f"{Path(tmp).stem}_{time.strftime('%Y%m%d')}"
            if stop == "edit" and urls:
                tpl = int(config.vizard_template_id) if config.vizard_template_id else None
                for c in clips:
                    if not c.url:
                        continue
                    st.write(f"Adding captions and cropping clip {c.rank}…")
                    try:
                        pid = submit_for_editing(config.vizard_api_key, c.url,
                                                 template_id=tpl,
                                                 project_name=f"#{c.rank}")
                        for done in parse_result(poll(config.vizard_api_key, pid,
                                                      verbose=False)):
                            # Vizard's links carry an Expires parameter and go
                            # dead in about a week, so the file is pulled down
                            # rather than the URL kept.
                            ready_dir.mkdir(parents=True, exist_ok=True)
                            safe = "".join(ch for ch in done.title
                                           if ch.isalnum() or ch in " -_")[:60]
                            local = ready_dir / f"{c.rank:02d} {safe.strip()}.mp4"
                            import requests as _rq
                            with _rq.get(done.video_url, stream=True,
                                         timeout=300) as resp:
                                resp.raise_for_status()
                                with local.open("wb") as fh:
                                    for chunk in resp.iter_content(1 << 16):
                                        fh.write(chunk)
                            link = ""
                            if not drive_on:
                                st.write("  (Drive not configured — local only)")
                            if drive_on:
                                # Uploaded from the local copy rather than
                                # streamed straight through, so a Drive failure
                                # cannot lose a clip we already paid to render.
                                try:
                                    st.write(f"  uploading {local.name} to Drive…")
                                    link = upload_drive(
                                        local, local.name,
                                        folder_id=config.drive_folder_id,
                                        creds_file=config.drive_creds,
                                        token_file=config.drive_token)
                                except StorageError as e:
                                    st.warning(f"Drive upload failed: {e}")
                            finished.append((c.rank, done.title, str(local), link))
                    except Exception as e:
                        st.warning(f"Clip {c.rank}: {e}")

            if stop == "edit" and not finished:
                st.warning("No finished clips came back from Vizard, so nothing "
                           "was delivered to Drive. The raw cuts are on disk.")
            status.update(label="Done", state="complete", expanded=False)

        # Survives the re-run that Streamlit triggers on the next interaction.
        st.session_state.result = {
            "meta": meta.model_dump(),
            "clips": [c.model_dump() for c in clips],
            "dir": str(clip_dir),
            "finished": finished,
            "ready_dir": str(ready_dir),
        }

    r = st.session_state.get("result")
    if r:
        st.divider()
        m = r["meta"]
        a, b, c_, d = st.columns(4)
        a.metric("Source", f"{m['duration'] / 60:.0f} min")
        b.metric("Clips", len(r["clips"]))
        c_.metric("Total", f"{sum(x['duration'] for x in r['clips']):.0f}s")
        d.metric("Kept", f"{sum(x['duration'] for x in r['clips']) / m['duration'] * 100:.1f}%")

        st.markdown(timeline(r["clips"], m["duration"]), unsafe_allow_html=True)

        # Eight video players stacked down the page is unreadable and slow to
        # load. The timeline above already says where each clip came from, so
        # the list stays text and a player opens only when asked for.
        for c in r["clips"]:
            st.markdown(
                f'<div class="clip"><div class="tc">{tc(c["start"])} → '
                f'{tc(c["end"])} · {c["duration"]:.0f}s · score {c["score"]} · '
                f'{c.get("topic") or ""}</div>'
                f'<div class="hook">{c["hook"]}</div></div>',
                unsafe_allow_html=True)
            p = Path(r["dir"]) / f"clip_{c['rank']:02d}.mp4"
            if p.exists():
                with st.expander(f"Watch the raw cut — clip {c['rank']}"):
                    st.video(str(p))
                    st.caption("Original framing, no captions. The finished "
                               "version is below.")


        if r["finished"]:
            st.divider()
            st.markdown("### Ready to post")
            st.caption(f"Local copy in `{r['ready_dir']}` — vertical, captioned, "
                       f"with the client's template applied.")
            if drive_on:
                st.markdown(f"**[Open the Drive folder]"
                            f"({folder_link(config.drive_folder_id)})** — send "
                            f"this link to whoever is scheduling.")
            for rank, title, path, link in r["finished"]:
                st.write(f"**{rank}. {title}**"
                         + (f" — [open on Drive]({link})" if link else ""))
                if Path(path).exists():
                    with st.expander(f"Watch clip {rank}"):
                        st.video(path)
            st.caption("Pick from these when scheduling. Anything weak just "
                       "does not get used — the spares are why there are more "
                       "than a week's worth.")

# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #

with tab_audit:
    st.markdown("# Where the client actually stands")
    st.caption("Their handle first, then whoever they think they compete with. "
               "Engagement is measured per follower, so the sizes stop mattering.")

    a1, a2 = st.columns([1, 2])
    client = a1.text_input("Client", "davanam_jewellers")
    rivals = a2.text_area(
        "Competitors, one per line",
        "abarantimelessjewellery\npraveenjewellers_bangalore\n"
        "lalithaajewellery_\nbhimajewelleryofficial\nb.b.jewellers",
        height=120)

    if config.competitor_backend == "apify" and not config.apify_token:
        st.warning("No APIFY_TOKEN. Add it to .env, or to Settings → Secrets "
                   "if this is deployed.")
        st.stop()

    if st.button("Pull last 25 posts each", type="primary"):

        handles = [client] + [h.strip().lstrip("@")
                              for h in rivals.splitlines() if h.strip()]
        with st.status(f"Pulling {len(handles)} accounts…", expanded=True) as s:
            if config.competitor_backend == "apify":
                st.write("One Apify run for all handles — a minute or two")
                comps = fetch_apify_batch(config.apify_token, handles, limit=25)
            else:
                comps = [fetch_competitor(config.ig_user_id, config.ig_access_token,
                                        h, limit=25) for h in handles]
            for c in comps:
                st.write(f"{'·' if c.error else '✓'} {c.username} "
                         f"{c.error[:60] if c.error else f'{c.followers:,} followers'}")
            path = save_competitor_snapshot(comps, Path(config.competitor_dir), client)
            s.update(label="Done", state="complete", expanded=False)
        st.session_state.audit = {"client": client, "snapshot": str(path)}

    au = st.session_state.get("audit")
    if au:

        out_dir = Path(config.competitor_dir)
        snaps = load_snapshots(out_dir, au["client"])
        if snaps:
            stamp, comps = snaps[-1]
            profile = load_profile(profile_path(au["client"], out_dir))
            html = _report.build(comps, au["client"], out_dir, stamp, profile)
            path = out_dir / f"{au['client']}_report_{stamp}.html"
            path.write_text(html, encoding="utf-8")

            st.divider()
            ok = [c for c in comps if not c.error and c.scorable]
            ok.sort(key=lambda c: c.median_rate(), reverse=True)
            for c in ok:
                mine = c.username.lower() == au["client"].lower()
                st.write(f"{'▸ ' if mine else ''}**@{c.username}** — "
                         f"{c.followers:,} followers · "
                         f"{c.median_rate() * 100:.2f}% engagement")

            st.download_button("Download the report", html,
                               file_name=path.name, mime="text/html")
            st.caption(f"Also written to {path}")


# --------------------------------------------------------------------------- #
# Track
#
# No account field here. Which account this reads is a .env decision — a token,
# an account id, and a Facebook Page linked behind it — and exposing it as a
# text box would suggest you can type any handle and get insights, which is
# exactly the thing the Graph API refuses to do.
#
# The metrics on this tab are the reason for going through Meta's setup at all.
# Scraping gets likes and comments. Reach, saves, shares and watch time exist
# only for the account owner, and they are the ones that say whether a clip was
# worth making.
# --------------------------------------------------------------------------- #

with tab_track:
    st.markdown("# What actually happened after posting")

    if not (config.client_ig_token and config.client_ig_user_id):
        missing = [k for k, v in (("CLIENT_IG_TOKEN", config.client_ig_token),
                                  ("CLIENT_IG_USER_ID", config.client_ig_user_id))
                   if not v]
        st.warning(f"Not connected to Instagram — missing {', '.join(missing)}.")
        st.caption(
            "Locally: run `python setup_meta.py YOUR_TOKEN`, which finds the "
            "account id and prints both values. Deployed: add them under "
            "Settings → Secrets, in TOML — `CLIENT_IG_TOKEN = \"EAA…\"`."
        )
        st.stop()

    tc1, tc2 = st.columns([1, 3])
    client_key = tc1.text_input("Label these results", "welvom",
                                help="Just a filename for the stored snapshots.")
    limit = tc2.select_slider("How many recent posts", [10, 25, 50, 100], value=25,
                              help="Insights cost one API call per post, against "
                                   "a limit of about 200 an hour. 25 is plenty "
                                   "for weekly tracking.")

    if st.button("Pull the numbers", type="primary"):

        with st.status("Reading Instagram…", expanded=True) as s_:
            stats = fetch_performance(config.client_ig_user_id,
                                   config.client_ig_token, limit=limit)
            if stats.error:
                s_.update(label="Failed", state="error")
                st.error(stats.error)
                st.stop()
            st.write(f"@{stats.username} · {stats.followers:,} followers · "
                     f"{len(stats.posts)} posts")
            reg = Path(config.performance_dir) / f"{client_key}_published.json"
            matched = attach_sources(stats, reg)
            if matched:
                st.write(f"{matched} of them came from this pipeline")
            path = save_performance_snapshot(stats, Path(config.performance_dir), client_key)
            s_.update(label="Done", state="complete", expanded=False)
        st.session_state.perf = {"client": client_key, "path": str(path)}

    # Show the last snapshot without being asked. Insights cost an API call per
    # post against a 200/hour ceiling, so re-pulling just to re-read numbers you
    # already have is wasteful — and looking at last week is the common case.
    pf = st.session_state.get("perf")
    if not pf:
        prev = sorted(Path(config.performance_dir)
                      .glob(f"{client_key}_performance_*.json"))
        if prev:
            pf = {"client": client_key, "path": str(prev[-1])}
            st.caption(f"Showing the snapshot from {prev[-1].stem[-8:]}. "
                       f"Pull again for current numbers.")
    if pf:

        stats = AccountStats(**json.loads(Path(pf["path"]).read_text(encoding="utf-8")))
        attach_sources(stats, Path(config.performance_dir)
                       / f"{pf['client']}_published.json")
        posts = [p for p in stats.posts if p.posted_at]
        posts.sort(key=lambda p: p.posted_at, reverse=True)

        st.divider()
        measured = [p for p in posts if p.reach]
        med_reach = (sorted(p.reach for p in measured)[len(measured) // 2]
                     if measured else 0)
        total_saves = sum(p.saved for p in posts)
        total_shares = sum(p.shares for p in posts)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Followers", f"{stats.followers:,}")
        k2.metric("Typical reach", f"{med_reach:,}",
                  help="Median across the posts measured")
        k3.metric("Saves", f"{total_saves:,}")
        k4.metric("Shares", f"{total_shares:,}")

        # The blunt number. Reach against your own follower count is the one
        # that tends to land, because most people assume their posts at least
        # get shown to the people who already follow them.
        if stats.followers and med_reach:
            pct = med_reach / stats.followers * 100
            msg = (f"A typical post reaches **{pct:.1f}%** of the "
                   f"{stats.followers:,} people who already follow this account.")
            (st.error if pct < 5 else st.info)(msg)

        # Format split. It is the same question the audit tab asks about
        # competitors, turned on ourselves.
        reels = [p for p in measured if p.is_reel]
        stills = [p for p in measured if not p.is_reel]
        if reels and stills:
            mr = sorted(p.reach for p in reels)[len(reels) // 2]
            ms = sorted(p.reach for p in stills)[len(stills) // 2]
            st.markdown(f"**Reels reach {mr:,}. Images reach {ms:,}.** "
                        f"{'Reels win by ' + format(mr / ms, '.1f') + '×.' if ms and mr > ms else ''}")
        elif stills and not reels:
            st.markdown(f"**Every post here is a still image.** "
                        f"Nothing to compare against yet — post one clip and this "
                        f"turns into a before and after.")

        st.divider()
        ours = [p for p in posts if p.clip_source]
        if ours:
            st.markdown("### What these clips did")
            st.caption("Ordered by save rate — saves per person reached. A save "
                       "means someone meant to come back to it, which is the "
                       "strongest signal a clip was worth making.")
            for p in sorted(ours, key=lambda x: -x.save_rate()):
                src = p.clip_source
                dur = float(src.get("duration") or 0)
                # One readable sentence rather than a row of fields. The point
                # is that a number is attached to an editorial decision.
                bits = [f"A **{dur:.0f}-second "
                        f"{str(src.get('content_type', 'clip')).replace('_', ' ')}"
                        f"** clip"]
                if src.get("topic"):
                    bits.append(f"about {str(src['topic']).replace('_', ' ')}")
                if src.get("hook"):
                    bits.append(f"opening on *\"{str(src['hook'])[:70]}…\"*")
                st.markdown(" ".join(bits))

                m1, m2, m3 = st.columns(3)
                m1.metric("Reach", f"{p.reach:,}")
                m2.metric("Save rate", f"{p.save_rate() * 100:.2f}%")
                if p.is_reel and p.avg_watch_s() and dur:
                    m3.metric("Watched through",
                              f"{p.avg_watch_s() / dur * 100:.0f}%",
                              help=f"{p.avg_watch_s():.1f}s of {dur:.0f}s")
                else:
                    m3.metric("Shares", f"{p.shares:,}")
                st.caption(f"Posted {p.posted_at[:10]} · "
                           f"[view]({p.permalink})")
                st.write("")
        else:
            st.info("No clips from this pipeline have been published yet. Once "
                    "one is, this is where its reach, saves and watch-through "
                    "appear.")

        if internal:
            with st.expander("Register a published clip"):
                st.caption("Do this once per clip, straight after posting. Without "
                           "it the numbers are just numbers — with it, next month's "
                           "clip selection can learn from them.")
                unreg = [p for p in posts if not p.clip_source]
                if unreg:
                    pick = st.selectbox(
                        "Which post", unreg,
                        format_func=lambda p: f"{p.posted_at[:10]} · "
                                              f"{'reel' if p.is_reel else 'image'} · "
                                              f"reach {p.reach:,} · {p.media_id}")
                    out_dir = Path(config.output_dir)
                    files = sorted(out_dir.glob("*_clips_*.json"),
                                   key=lambda f: -f.stat().st_mtime)
                    if files:
                        cf = st.selectbox("Which run produced it", files,
                                          format_func=lambda f: f.name)
                        clips = json.loads(cf.read_text(encoding="utf-8"))
                        ck = st.selectbox(
                            "Which clip", clips,
                            format_func=lambda c: f"#{c['rank']} · {c['duration']:.0f}s "
                                                  f"· {c.get('topic') or ''} · "
                                                  f"{c['hook'][:50]}")
                        if st.button("Link them"):
                            ck["source_video"] = cf.stem
                            register(Path(config.performance_dir)
                                     / f"{pf['client']}_published.json",
                                     pick.media_id, ck, pick.permalink)
                            st.success("Linked. Pull the numbers again to see it.")
                    else:
                        st.caption("No clip runs found in data/output yet.")
                else:
                    st.caption("Every post is already registered.")

        st.divider()
        st.markdown("### Every post")
        st.caption("Views counts every appearance including repeat viewers; "
                   "reach counts people once. Saves and shares are the ones "
                   "scraping cannot see.")
        rows = [{
            "Posted": p.posted_at[:10],
            "Kind": "reel" if p.is_reel else p.media_type.lower(),
            "Views": p.views,
            "Reach": p.reach,
            "Likes": p.likes,
            "Saves": p.saved,
            "Shares": p.shares,
            "Save %": round(p.save_rate() * 100, 2),
            "From pipeline": p.clip_source.get("topic", "") if p.clip_source else "",
        } for p in posts]
        st.dataframe(rows, width="stretch", hide_index=True)

        snaps = sorted(Path(config.performance_dir)
                       .glob(f"{pf['client']}_performance_*.json"))
        st.caption(f"{len(snaps)} snapshot(s) stored. Run this weekly — a post's "
                   f"numbers keep moving for days, and Instagram will not tell "
                   f"you later what last week's reach was.")