#!/usr/bin/env python3
"""Step 4 — send pre-cut clips to an editor for captions and vertical crop.

Two providers, same command:

    --provider vizard     cheaper (~$0.17/clip), generic AI headline
    --provider submagic   ~$0.41-0.69/clip, but takes OUR hook text and OUR
                          B-roll placement

Both need a PUBLIC URL, not a local file, so upload your cut clips somewhere
first. Run --storage-help for the options.

    # one clip through each, to compare
    python edit.py --provider vizard   --url https://.../clip_01.mp4
    python edit.py --provider submagic --url https://.../clip_01.mp4 \
                   --hook "Three weeks in, people ask what you're on"

    # every clip from a run, given the URL prefix they were uploaded under
    python edit.py --provider submagic --clips data/output/<id>_clips_G.json \
                   --prefix https://your-bucket.r2.dev/<id>

    python edit.py --list-templates --provider submagic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.config import config
from pipeline.vizard import README_STORAGE


def run_vizard(args, urls) -> int:
    from pipeline.vizard import VizardError, parse_result, poll, submit_for_editing

    if not config.vizard_api_key:
        print("VIZARDAI_API_KEY is not set in .env", file=sys.stderr)
        return 1
    tpl = int(config.vizard_template_id) if config.vizard_template_id else None
    failures = 0

    for label, url in urls:
        print(f"\n{label}\n  submitting {url}")
        try:
            pid = submit_for_editing(
                config.vizard_api_key, url,
                video_type=3 if args.gdrive else 1,
                headline=not args.no_headline,
                auto_broll=args.broll,
                remove_silence=args.remove_silence,
                template_id=tpl,
                project_name=label,
            )
            print(f"  accepted, project {pid}")
            result = poll(config.vizard_api_key, pid)
            print(f"  credits used: {result.get('creditsUsed')}")
            for c in parse_result(result):
                print(f"\n  READY  {c.duration_s}s")
                print(f"  headline : {c.title}")
                if c.topics:
                    print(f"  topics   : {', '.join(c.topics)}")
                print(f"  download : {c.video_url}")
        except VizardError as e:
            failures += 1
            print(f"  FAILED: {e}", file=sys.stderr)
        except Exception as e:
            failures += 1
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    return 1 if failures else 0


def run_submagic(args, urls, texts) -> int:
    from pipeline.submagic import SubmagicError, create_project, poll

    if not config.submagic_api_key:
        print("SUBMAGIC_API_KEY is not set in .env", file=sys.stderr)
        return 1

    # Write the hooks up front, so a failure here costs no Submagic credits.
    hooks: dict[str, str] = {}
    if args.write_hooks:
        from pipeline.hooks import write_hook

        print("\nwriting hooks:")
        for label, _ in urls:
            text = texts.get(label, "")
            if not text:
                print(f"  {label}: no transcript in the clips file, skipping")
                continue
            hk, thin, why = write_hook(text, content_type=args.type,
                                       brand_note=args.brand)
            hooks[label] = hk.overlay
            flag = f"   (spoken opening thin: {why})" if thin else ""
            print(f"  {label}  {hk.overlay}{flag}")
            if hk.caption_open:
                print(f"      caption: {hk.caption_open[:110]}")

    failures = 0
    for label, url in urls:
        print(f"\n{label}\n  submitting {url}")
        try:
            pid = create_project(
                config.submagic_api_key, url,
                title=label,
                minimal=args.minimal,
                template_name=config.submagic_template or None,
                preset_id=config.submagic_preset or None,
                hook_text=hooks.get(label, args.hook),
                auto_hook=args.auto_hook,
                magic_zooms=not args.no_zooms,
                magic_brolls=args.broll,
                silence_pace=args.silence,
                remove_bad_takes=args.bad_takes,
                dictionary=[d.strip() for d in args.dictionary.split(",") if d.strip()],
            )
            print(f"  accepted, project {pid}")
            p = poll(config.submagic_api_key, pid)
            print(f"\n  READY")
            print(f"  download : {p.download_url}")
        except SubmagicError as e:
            failures += 1
            print(f"  FAILED: {e}", file=sys.stderr)
        except Exception as e:
            failures += 1
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="vizard", choices=("vizard", "submagic"))
    ap.add_argument("--url", help="single clip URL")
    ap.add_argument("--clips", type=Path, help="the *_clips_*.json from clip.py")
    ap.add_argument("--prefix", help="public URL prefix the clips were uploaded under")
    ap.add_argument("--gdrive", action="store_true",
                    help="vizard only: the URL is a Google Drive share link")

    ap.add_argument("--hook", default="",
                    help="submagic only: one hook text, used for every clip")
    ap.add_argument("--minimal", action="store_true",
                    help="submagic only: send just title, language and videoUrl, "
                         "to isolate which option a 400 is complaining about")
    ap.add_argument("--auto-hook", action="store_true",
                    help="submagic only: let Submagic write the hook itself")
    ap.add_argument("--write-hooks", action="store_true",
                    help="submagic only: have Claude write a hook per clip from "
                         "its transcript, then pass it to Submagic")
    ap.add_argument("--type", default="interview",
                    help="content type, passed to the hook writer")
    ap.add_argument("--brand", default="",
                    help="one line on brand voice, passed to the hook writer")
    ap.add_argument("--no-headline", action="store_true",
                    help="vizard only: turn off its AI headline")
    ap.add_argument("--no-zooms", action="store_true",
                    help="submagic only: disable magic zooms")
    ap.add_argument("--broll", action="store_true", help="enable auto B-roll")
    ap.add_argument("--silence", default="off",
                    choices=("off", "natural", "fast", "extra-fast"),
                    help="submagic only: silence removal pace")
    ap.add_argument("--remove-silence", action="store_true",
                    help="vizard only: strip silence and filler (default off; "
                         "their docs warn it can make clips choppy)")
    ap.add_argument("--bad-takes", action="store_true",
                    help="submagic only: remove bad takes")
    ap.add_argument("--dictionary", default="",
                    help="submagic only: comma-separated proper nouns, e.g. "
                         "'Raj Shamani,WelvomAI' — fixes recurring misheards")

    ap.add_argument("--list-templates", action="store_true")
    ap.add_argument("--storage-help", action="store_true")
    args = ap.parse_args()

    if args.storage_help:
        print(README_STORAGE)
        return 0

    if args.list_templates:
        if args.provider == "submagic":
            from pipeline.submagic import list_hook_templates, list_templates
            print("caption templates:")
            for t in list_templates(config.submagic_api_key):
                print(f"  {t}")
            print("\nhook templates:")
            for t in list_hook_templates(config.submagic_api_key):
                print(f"  {t}")
        else:
            print("Vizard template IDs come from the editor UI: "
                  "open a project, Template tab, hover a template.")
        return 0

    urls: list[tuple[str, str]] = []
    texts: dict[str, str] = {}
    if args.url:
        urls.append(("clip", args.url))
    elif args.clips:
        clips = json.loads(args.clips.read_text(encoding="utf-8"))
        for c in clips:
            label = f"#{c['rank']} {c.get('topic') or ''}"
            # A clips file written by `clip.py --publish` already carries the
            # URL; --prefix is only needed when the clips were uploaded by hand.
            if c.get("url"):
                urls.append((label, c["url"]))
            elif args.prefix:
                urls.append((label, f"{args.prefix.rstrip('/')}/clip_{c['rank']:02d}.mp4"))
            else:
                print(f"  #{c['rank']} has no url — run clip.py --publish, "
                      f"or pass --prefix", file=sys.stderr)
                continue
            texts[label] = c.get("text", "")
    else:
        print("Give --url, or --clips (with --prefix if not published)",
              file=sys.stderr)
        return 1

    print(f"provider: {args.provider}")
    if args.write_hooks and not texts:
        print("--write-hooks needs --clips (the transcript lives in that file)",
              file=sys.stderr)
        return 1

    if args.provider == "submagic":
        return run_submagic(args, urls, texts)
    return run_vizard(args, urls)


if __name__ == "__main__":
    sys.exit(main())