"""The half of an audit that cannot be scraped.

WHY THIS IS A FILE RATHER THAN A PROMPT

Reading two hand-written audits made the split obvious. Roughly half of what is
in them comes from data already sitting in the competitor snapshots — follower
counts, per-post engagement, posting gaps, hashtags, which formats win. The
report was simply not showing it.

The other half cannot be derived from any API: that 80-90% of Davanam's revenue
is word-of-mouth, that every lead lands on Vijay's personal phone, that prices
must never appear in a post, that the ICP is 30-55 and buys for weddings. That
comes from the onboarding call.

So it goes in a file per client, written once, and the report merges the two.
Everything here is optional — a profile with only a handle still produces a
report, just a thinner one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


class ReelIdea(BaseModel):
    idea: str
    goal: str = ""          # what we are trying to achieve with it


class ClientProfile(BaseModel):
    # --- identity ---
    name: str = ""
    handle: str = ""                    # their own Instagram, tracked alongside
    facebook_followers: int = 0         # platforms we cannot pull
    established: str = ""               # "Since 1905"
    locations: list[str] = Field(default_factory=list)

    # --- how the business actually works ---
    revenue_note: str = ""              # "80-90% referral / word-of-mouth"
    lead_routing: str = ""              # "every lead lands on one personal phone"
    moat: str = ""                      # what they actually sell on

    # --- what we found ---
    pain_points: list[str] = Field(default_factory=list)
    their_plans: list[str] = Field(default_factory=list)   # said on the call

    # --- constraints ---
    hard_rules: list[str] = Field(default_factory=list)    # from onboarding form
    brand_colours: str = ""
    tagline: str = ""

    # --- who we are talking to ---
    icp: list[str] = Field(default_factory=list)
    goal: str = ""
    cadence: str = ""                   # "10 posts/month: 5 images + 5 reels"

    # --- what we do about it ---
    reel_ideas: list[ReelIdea] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)

    # --- hashtags we recommend, grouped the way they get used ---
    hashtags_local: list[str] = Field(default_factory=list)
    hashtags_service: list[str] = Field(default_factory=list)
    hashtags_broad: list[str] = Field(default_factory=list)


def load_profile(path: Path) -> Optional[ClientProfile]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return ClientProfile(**json.loads(p.read_text(encoding="utf-8")))
    except (ValueError, TypeError) as e:
        print(f"  profile at {p} could not be read: {e}")
        return None


def profile_path(client: str, base: Path) -> Path:
    return Path(base) / f"{client}_profile.json"


TEMPLATE: dict[str, Any] = {
    "name": "Davanam Jewellers",
    "handle": "davanam_jewellers",
    "facebook_followers": 24000,
    "established": "Quality & Trust Since 1905",
    "locations": ["Commercial Street", "Malleshwaram", "Avenue Road"],

    "revenue_note": "80-90% of revenue is referral and word-of-mouth. The referral "
                    "engine is product-led, not ask-led: a customer wears the piece, "
                    "someone compliments it, the conversation starts.",
    "lead_routing": "Every lead lands on one personal phone. WhatsApp, calls and DMs "
                    "all route to the same number, deliberately.",
    "moat": "Four generations, IGI certification, 40,000 satisfied customers, and a "
            "15-year Divine Solitaires partnership. The content promotes products; "
            "the moat is trust.",

    "pain_points": [
        "No consistency in posting",
        "Not enough engagement",
        "Does not appear when searching 'Bangalore jewellers'",
        "The wrong asset is being promoted — content sells products, the moat is trust",
        "Traffic campaigns optimise for profile visits, not buyers",
    ],
    "their_plans": [
        "Hook, product showcase, contact details",
        "Event posts (Friendship Day, Independence Day)",
        "Owner snippets on the Davanam legacy",
        "Daily-wear diamonds and lightweight men's collections",
        "Festive offer announcements",
    ],

    "hard_rules": [
        "Never show product prices in posts",
        "Client names and images need written approval before use",
        "Management approves every post before publishing",
    ],
    "brand_colours": "Maroon and Gold",
    "tagline": "Quality & Trust Since 1905",

    "icp": [
        "Bridal and milestone buyers, plus high-LTV repeat customers",
        "Age 30-55",
        "Business owners, professionals, government employees",
        "Active on Facebook and Instagram",
    ],
    "goal": "Brand awareness, follower growth, and measurable leads. "
            "Stated target: 30% of revenue from digital.",
    "cadence": "Instagram and Facebook only. 10 posts a month each: 5 images, 5 reels.",

    "reel_ideas": [
        {"idea": "Emotional family narrative", "goal": "Built to be forwarded"},
        {"idea": "Person in showroom wearing and explaining a piece",
         "goal": "Generates the price comment"},
        {"idea": "Recurring owner face", "goal": "Brand recall"},
        {"idea": "Local Kannada creator or comedy skit", "goal": "Mass local reach"},
        {"idea": "Customer testimonial", "goal": "Trust building"},
    ],
    "deliverables": [
        "Reel posting",
        "Hashtags, captions, geotags",
        "DM automation",
        "Cross-posting to Facebook",
        "Auto-scheduling",
        "Attribution capture — QR at point of sale asking how they found us",
        "Customer segmentation and targeted follow-up messages",
    ],
    "open_questions": [
        "Is comment-to-DM automated, or is someone replying by hand?",
        "The scroller and the buyer are not the same person. Bridal jewellery is a "
        "group decision — who are we actually writing for?",
        "Nothing currently measures whether a walk-in came from social.",
    ],

    "hashtags_local": ["#BangaloreJewellers", "#CommercialStreet", "#Malleshwaram"],
    "hashtags_service": ["#BridalJewellery", "#TempleJewellery", "#DiamondJewellery"],
    "hashtags_broad": ["#GoldJewellery", "#IndianJewellery", "#JewelleryDesign"],
}


def write_template(path: Path) -> Path:
    """Drop a filled-in example next to the client's data."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(TEMPLATE, indent=2, ensure_ascii=False), encoding="utf-8")
    return p