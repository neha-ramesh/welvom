"""Write the on-screen hook text for a finished clip.

WHAT THIS DOES AND DELIBERATELY DOES NOT DO

It reads a clip's transcript and returns 3-8 words to burn over the first two
seconds, plus an opening line for the social caption. That is all.

An earlier version of this fed its analysis back into the boundary selection —
if a clip opened on a word that referred to something unseen, the boundary pass
was told to start earlier and name the subject. That made every clip open with
preamble instead of the hook, and the output got measurably worse: starts moved
2-3 seconds earlier and durations drifted up.

So the orphan check stays here, as CONTEXT FOR WRITING THE OVERLAY, and never
reaches the boundary logic. The division is:

    boundary pass   picks the punchiest defensible opening
    this module     supplies whatever context that opening lacks

The reason that works is that on-screen text is read faster than speech is
heard. In two seconds a viewer reads six words but hears about four. So the
overlay stops the scroll and the audio delivers the payload, which means a
slightly context-free spoken opening is recoverable — as long as the overlay
knows it has to do that job.
"""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel

from .llm import complete_json

# Words that point at something previously said. An opening led by one of these,
# with no noun to anchor it, leaves the viewer guessing.
_ORPHANS = {
    "that", "that's", "thats", "this", "these", "those", "it", "it's", "its",
    "they", "them", "their", "he", "she", "his", "her", "him", "there",
    "there's", "theres",
}
_CONTINUATIONS = {
    "so", "and", "but", "because", "or", "then", "also", "however",
    "therefore", "anyway", "plus", "though", "although", "which", "right",
}


class HookSet(BaseModel):
    overlay: str = ""        # burned over the first ~2s, 3-8 words
    caption_open: str = ""   # opening line of the social post
    why: str = ""


SCHEMA = {
    "type": "object",
    "properties": {
        "overlay": {"type": "string"},
        "caption_open": {"type": "string"},
        "why": {"type": "string"},
    },
    "required": ["overlay", "caption_open"],
}


SYSTEM_PROMPT = """You write the on-screen hook for short-form video clips.

The overlay is text burned over the first two seconds. A viewer reads it before
they have heard a full sentence, so it is what actually stops the scroll.

Rules for the overlay:
- 3 to 8 words. Shorter is stronger. It has to land at a glance.
- Make the viewer need the answer: a claim they want to test, a number that
  seems wrong, a question they have privately asked themselves.
- It must be TRUE to what is actually said. Never promise something the clip
  does not deliver — that is what makes people leave annoyed instead of
  watching again.
- Never a label. "Gratitude and the immune system" is a chapter title.
  "10 minutes of gratitude beats 10 minutes of Instagram" is a hook.
- No hashtags, no emoji, no "watch till the end", no "you won't believe".
- Sentence case. No trailing full stop.

You will be told whether the clip's SPOKEN opening stands on its own.

If it does NOT, the overlay has a second job: name the subject the speaker
never does. A clip that opens "That's the joy of change" leaves the viewer
asking what "that" is, so the overlay must tell them — something like "People
asked if he was on a new drug". Now the line has a referent and the clip works.

If the spoken opening DOES stand alone, don't repeat it. Sharpen it instead:
add the number, or pose the question the line answers.

Also write caption_open: the first line or two of the social post,
conversational, no hashtags. It can be fuller than the overlay because people
read captions after they have already stopped scrolling.

Return ONLY JSON: {"overlay": str, "caption_open": str, "why": "one line"}"""


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def first_sentence(text: str, max_words: int = 24) -> str:
    m = re.search(r"^(.{5,}?[.?!])(\s|$)", text.strip())
    s = m.group(1) if m else text.strip()
    return " ".join(s.split()[:max_words])


def needs_context(text: str) -> tuple[bool, str]:
    """Does this opening depend on something the viewer has not heard?

    Used ONLY to brief the hook writer. It never influences where a clip starts.
    """
    toks = _tokens(text)
    if not toks:
        return True, "empty opening"
    head = toks[0]

    if head in _CONTINUATIONS:
        rest = toks[1:7]
        if not rest:
            return True, f"opens on '{head}' with nothing after it"
        if rest[0] in _ORPHANS:
            return True, f"opens on '{head} {rest[0]}' — no subject named"
        return True, f"opens mid-argument on '{head}'"

    if head in _ORPHANS:
        # "That's the joy of change" is a bare pronoun and orphans the viewer.
        # "That immune response is why…" is a determiner with its noun attached,
        # so the subject is present and it reads fine.
        if head in {"that's", "thats", "it's", "its", "there's", "theres"}:
            return True, f"opens on '{head}' — no subject named"
        nxt = toks[1] if len(toks) > 1 else ""
        if head in {"that", "this", "these", "those", "their", "his", "her"} and (
            len(nxt) > 3 and nxt not in _ORPHANS and nxt not in _CONTINUATIONS
        ):
            return False, ""
        return True, f"opens on '{head}' — refers to something unseen"

    return False, ""


def write_hook(
    clip_text: str,
    *,
    topic: Optional[str] = None,
    content_type: str = "interview",
    brand_note: str = "",
    verbose: bool = True,
) -> tuple[HookSet, bool, str]:
    """Generate overlay and caption for one clip.

    Returns (hooks, spoken_opening_needs_context, reason).
    """
    opening = first_sentence(clip_text)
    thin, reason = needs_context(opening)

    lines = [
        f"CLIP TRANSCRIPT:\n{clip_text[:1800]}",
        "",
        f'SPOKEN OPENING: "{opening}"',
        (
            f"This opening does NOT stand on its own: {reason}. "
            "The overlay must name the missing subject."
            if thin
            else "This opening stands on its own. Sharpen it, do not repeat it."
        ),
    ]
    if topic:
        lines.append(f"Topic: {topic}")
    if content_type:
        lines.append(f"Content type: {content_type}")
    if brand_note:
        lines.append(f"Brand voice: {brand_note}")

    try:
        data = complete_json(
            SYSTEM_PROMPT,
            "\n".join(lines),
            schema=SCHEMA,
            list_key="hook",
            required_field="overlay",
            temperature=0.4,   # hooks benefit from a little variety
        )
        hooks = HookSet(**data)
    except Exception as e:
        if verbose:
            print(f"      hook: generation failed ({e})")
        return HookSet(), thin, reason

    words = hooks.overlay.split()
    if len(words) > 10:
        hooks.overlay = " ".join(words[:10])
    hooks.overlay = hooks.overlay.strip().rstrip(".")
    return hooks, thin, reason