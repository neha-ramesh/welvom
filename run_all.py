#!/usr/bin/env python3
"""Video in, finished reels out. One command.

    python run_all.py data/input/podcast.mp4 --type interview
    python run_all.py data/input/podcast.mp4 --type testimonial -n 3
    python run_all.py data/input/podcast.mp4 --stop-after cut

Runs the five stages in order: transcribe, pick clips, cut, upload, edit.

RESUMABLE BY DEFAULT

Transcription is the slow, paid stage — a 2-hour video is several minutes and
three API calls. So if the analysis JSON already exists it is reused, and only
the stages after it re-run. That matters because the stages most likely to fail
are the ones at the end (a storage token, a Vizard credit limit), and re-running
should not mean paying for the transcript again.

Pass --fresh to force it.

WHERE IT STOPS ON PURPOSE

--stop-after cut leaves you with local mp4s and no uploads. Use that while you
are still judging whether the clips are good, because publishing and editing
both cost money and neither improves a badly chosen clip.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pipeline.build import analyse_video, save
from pipeline.config import config
from pipeline.llm import describe_backend
from pipeline.models import VideoAnalysis
from pipeline.probe import extract_metadata
from pipeline.select import PLAYBOOKS, select_clips
from pipeline.storage import StorageError, upload_clips

STAGES = ("transcribe", "select", "cut", "publish", "edit")


def banner(n: int, total: int, text: str) -> None:
    print(f"\n[{n}/{total}] {text}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--type", default="generic", choices=sorted(PLAYBOOKS))
    ap.add_argument("-n", type=int, default=5, help="how many clips")
    ap.add_argument("--mode", default="direct", choices=("segments", "direct"))
    ap.add_argument("--llm", choices=("gemini", "claude"), default="claude")
    ap.add_argument("--tag", default="auto")
    ap.add_argument("--stop-after", choices=STAGES, default="edit")
    ap.add_argument("--fresh", action="store_true",
                    help="re-transcribe even if an analysis file exists")
    ap.add_argument("--broll", action="store_true", help="Vizard auto B-roll")
    ap.add_argument("--emoji", action="store_true",
                    help="Vizard auto emoji in captions (their default is off)")
    ap.add_argument("--no-headline", action="store_true",
                    help="turn off Vizard's AI headline")
    args = ap.parse_args()

    if not args.video.exists():
        print(f"{args.video} does not exist", file=sys.stderr)
        return 1

    if args.llm == "claude":
        config.llm_backend = "anthropic"
    else:
        config.llm_backend = "openai"

    stop_at = STAGES.index(args.stop_after)
    out_dir = Path(config.output_dir)
    t0 = time.time()

    # ---------------------------------------------------------------- 1. transcribe
    # The video id is a hash of the file, so the analysis path is knowable
    # before doing any work — which is what makes the resume check possible.
    meta = extract_metadata(args.video)
    analysis_path = out_dir / f"{meta.video_id}.json"

    banner(1, 5, f"transcribe  ({meta.duration / 60:.1f} min)")
    if analysis_path.exists() and not args.fresh:
        analysis = VideoAnalysis(**json.loads(analysis_path.read_text(encoding="utf-8")))
        print(f"      reusing {analysis_path.name} "
              f"({len(analysis.transcript.segments)} chunks) — --fresh to redo")
    else:
        analysis = analyse_video(args.video)
        save(analysis, out_dir)
        print(f"      wrote {analysis_path.name}")
    if stop_at == 0:
        return 0

    # ---------------------------------------------------------------- 2. select
    banner(2, 5, f"pick clips  ({args.mode} mode, {describe_backend()})")
    clips = select_clips(analysis, content_type=args.type, top_n=args.n,
                         mode=args.mode)
    if not clips:
        print("      no clips survived the filters", file=sys.stderr)
        return 1
    for c in clips:
        print(f"      #{c.rank}  {c.start:.0f}-{c.end:.0f}s ({c.duration:.0f}s)  "
              f"score {c.score}  {c.hook[:60]}")
    if stop_at == 1:
        return 0

    # ---------------------------------------------------------------- 3. cut
    from clip import DECLICK_S, render  # reuse the renderer rather than fork it
    from pipeline.boundary import neighbours, refine

    banner(3, 5, "cut")
    words = analysis.transcript.words
    for c in clips:
        lo, hi = neighbours(words, c.start, "start")
        c.start, _ = refine(args.video, c.start, lower=lo, upper=hi)
        lo, hi = neighbours(words, c.end, "end")
        c.end, _ = refine(args.video, c.end, lower=lo, upper=hi)
        c.duration = round(c.end - c.start, 2)

    tag = args.tag if args.tag != "auto" else args.type
    clip_dir = out_dir / f"{meta.video_id}_clips_{tag}"
    for c in clips:
        print(f"      {render(args.video, c, clip_dir).name}")

    clips_json = out_dir / f"{meta.video_id}_clips_{tag}.json"

    def write_clips() -> None:
        clips_json.write_text(
            json.dumps([c.model_dump() for c in clips], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    write_clips()
    if stop_at == 2:
        print(f"\nStopped after cut. Watch {clip_dir} before spending on the rest.")
        return 0

    # ---------------------------------------------------------------- 4. publish
    banner(4, 5, f"publish  ({config.storage_backend})")
    try:
        urls = upload_clips(clip_dir, meta.video_id, config)
    except StorageError as e:
        print(f"      FAILED: {e}", file=sys.stderr)
        print(f"      Clips are still on disk at {clip_dir}", file=sys.stderr)
        return 1
    for c in clips:
        c.url = urls.get(c.rank, "")
    write_clips()   # so a later edit.py run can read the URLs straight out
    if stop_at == 3:
        return 0

    # ---------------------------------------------------------------- 5. edit
    banner(5, 5, "edit  (Vizard)")
    if not config.vizard_api_key:
        print("      VIZARDAI_API_KEY is not set — clips are published and ready",
              file=sys.stderr)
        print(f"      resume with: python edit.py --clips {clips_json}",
              file=sys.stderr)
        return 1

    from pipeline.vizard import VizardError, parse_result, poll, submit_for_editing

    tpl = int(config.vizard_template_id) if config.vizard_template_id else None
    failed = 0
    for c in clips:
        if not c.url:
            continue
        print(f"\n      #{c.rank} {c.topic or ''}")
        try:
            pid = submit_for_editing(
                config.vizard_api_key, c.url,
                headline=not args.no_headline,
                auto_broll=args.broll,
                emoji=args.emoji,
                template_id=tpl,
                project_name=f"#{c.rank} {c.topic or meta.video_id}",
            )
            for done in parse_result(poll(config.vizard_api_key, pid)):
                print(f"      ready: {done.title}")
                print(f"      {done.video_url}")
        except (VizardError, Exception) as e:
            failed += 1
            print(f"      FAILED: {e}", file=sys.stderr)

    print(f"\ndone in {time.time() - t0:.0f}s"
          f"{f' ({failed} clip(s) failed at the edit stage)' if failed else ''}")
    print(f"clips file: {clips_json}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())