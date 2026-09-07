#!/usr/bin/env python3
"""Step 3 CLI — pick the best clips from a Step 2 analysis JSON.

    python clip.py data/output/51f11977cfc8.json --type founder_talk
    python clip.py data/output/51f11977cfc8.json --type testimonial -n 5
    python clip.py data/output/51f11977cfc8.json --cut      # also render mp4s
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pipeline.boundary import neighbours, refine
from pipeline.config import config
from pipeline.storage import StorageError, upload_clips
from pipeline.edges import optimise_edges
from pipeline.llm import describe_backend
from pipeline.models import VideoAnalysis
from pipeline.select import PLAYBOOKS, select_clips


# A short audio ramp at each edge. Separate from boundary refinement and not a
# substitute for it: refinement picks a better cut point, the fade makes
# whatever point we picked sound soft instead of popping. Both run.
DECLICK_S = 0.02


def render(video: Path, clip, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"clip_{clip.rank:02d}.mp4"
    fade_out_at = max(0.0, clip.duration - DECLICK_S)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(clip.start), "-i", str(video),
         "-t", str(clip.duration),
         "-af", f"afade=t=in:st=0:d={DECLICK_S},"
                f"afade=t=out:st={fade_out_at:.3f}:d={DECLICK_S}",
         "-c:v", "libx264", "-preset", "veryfast",
         "-c:a", "aac", "-loglevel", "error", str(out)],
        check=True,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("analysis", type=Path, help="Step 2 JSON")
    ap.add_argument("--type", default="generic", choices=sorted(PLAYBOOKS))
    ap.add_argument("--mode", default="segments", choices=("segments", "direct"),
                    help="segments = use topic groups; direct = read chunks straight")
    ap.add_argument("--llm", choices=("gemini", "claude"),
                    help="override LLM_BACKEND for this run")
    ap.add_argument("--tag", default="",
                    help="suffix for the output filename, for comparing runs")
    ap.add_argument("--publish", action="store_true",
                    help="after cutting, upload the clips and record their public "
                         "URLs in the clips file (needs --cut)")
    ap.add_argument("--no-refine", action="store_true",
                    help="cut exactly at the chosen timestamps, no DSP refinement")
    ap.add_argument("--no-edges", action="store_true",
                    help="skip the boundary optimiser LLM pass")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--cut", type=Path, default=None,
                    help="source video; also renders the clips as mp4")
    args = ap.parse_args()

    if args.llm == "claude":
        config.llm_backend = "anthropic"
    elif args.llm == "gemini":
        config.llm_backend = "openai"

    analysis = VideoAnalysis(**json.loads(args.analysis.read_text(encoding="utf-8")))
    if args.mode == "segments" and not analysis.semantic_segments:
        print("No semantic_segments — run Step 2 without --no-segment first.",
              file=sys.stderr)
        return 1

    print(f"chunks={len(analysis.transcript.segments)} "
          f"segments={len(analysis.semantic_segments)} "
          f"mode={args.mode} playbook={args.type}")
    print(f"model: {describe_backend()}")
    clips = select_clips(
        analysis, content_type=args.type, top_n=args.n, mode=args.mode
    )

    if not clips:
        print("No candidates survived the duration and overlap filters.")
        return 0

    # Pass 2: a dedicated model call picks the exact opening and closing lines.
    # Runs before the DSP pass, which then finds the acoustic instant beside the
    # word this chose. Order matters: content, then edges, then samples.
    if not args.no_edges:
        from pipeline.select import TARGET_MAX, TARGET_MIN
        kept = []
        for c in clips:
            ns, ne, complete, why = optimise_edges(
                analysis.transcript, c.start, c.end,
                min_s=TARGET_MIN, max_s=TARGET_MAX, hook=c.hook,
            )
            moved = abs(ns - c.start) > 0.05 or abs(ne - c.end) > 0.05
            if not complete:
                # Reject rather than downscore. A clip that opens on a fragment
                # or stops before the point lands is not worth publishing, and
                # a lower score would still let it through as rank 5.
                print(f"  #{c.rank} REJECTED by completeness gate: {why}")
                continue
            if moved:
                print(f"  #{c.rank} edges: {c.start:.1f}->{ns:.1f}s  "
                      f"{c.end:.1f}->{ne:.1f}s  ({why[:60]})")
            c.start, c.end = ns, ne
            c.duration = round(ne - ns, 2)
            kept.append(c)
        clips = kept
        for i, c in enumerate(clips, 1):
            c.rank = i
        if not clips:
            print("Every candidate failed the completeness gate.", file=sys.stderr)
            return 1

    # Refine every boundary against the real audio. Needs the media file, so it
    # only runs when --cut was given.
    if args.cut and not args.no_refine:
        words = analysis.transcript.words
        for c in clips:
            lo, hi = neighbours(words, c.start, "start")
            ns, moved_s = refine(args.cut, c.start, lower=lo, upper=hi)
            lo, hi = neighbours(words, c.end, "end")
            ne, moved_e = refine(args.cut, c.end, lower=lo, upper=hi)
            if moved_s or moved_e:
                print(f"  #{c.rank} refined: start {c.start:.3f}->{ns:.3f} "
                      f"end {c.end:.3f}->{ne:.3f}")
            c.start, c.end = ns, ne
            c.duration = round(ne - ns, 2)

    for c in clips:
        flag = "" if c.snapped_to_sentence else "   (no sentence edge found)"
        print(f"\n#{c.rank}  {c.start:.1f}s - {c.end:.1f}s  ({c.duration:.0f}s)  "
              f"score {c.score}  [{c.topic or '-'}]{flag}")
        print(f"    opens: {c.text[:90]}")
        print(f"     hook: {c.hook[:90]}")
        print(f"      why: {c.reason[:100]}")

    tag = f"_{args.tag}" if args.tag else f"_{args.mode}"
    out = args.analysis.parent / f"{analysis.metadata.video_id}_clips{tag}.json"
    out.write_text(
        json.dumps([c.model_dump() for c in clips], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")

    if args.cut:
        d = args.analysis.parent / f"{analysis.metadata.video_id}_clips{tag}"
        for c in clips:
            print(f"  rendered {render(args.cut, c, d).name}")

        if args.publish:
            print("\npublishing:")
            try:
                urls = upload_clips(d, analysis.metadata.video_id, config)
                for c in clips:
                    c.url = urls.get(c.rank, "")
                # Rewrite the clips file so edit.py can read the URLs straight
                # out of it rather than being handed a prefix by hand.
                out.write_text(
                    json.dumps([c.model_dump() for c in clips], indent=2,
                               ensure_ascii=False),
                    encoding="utf-8",
                )
                for c in clips:
                    if c.url:
                        print(f"  #{c.rank}  {c.url}")
                print(f"\nnext:  python edit.py --provider vizard --clips {out}")
            except StorageError as e:
                print(f"  FAILED: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())