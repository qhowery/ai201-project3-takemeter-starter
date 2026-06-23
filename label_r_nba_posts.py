#!/usr/bin/env python3
"""Label r/nba posts for TakeMeter Milestone 3 using planning.md definitions."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

VALID_LABELS = {"analysis", "hot_take", "news_updates", "meta_community"}

# Manual overrides for genuinely ambiguous edge cases only.
OVERRIDES: dict[str, str] = {
    "1kqa94r": "meta_community",
    "1kq80bg": "meta_community",
    "1kq73rb": "meta_community",
    "1kq3e0b": "meta_community",
    "1kq2c4v": "meta_community",
    "1kq1cx7": "meta_community",
    "1kq19el": "hot_take",
    "1kq10yo": "news_updates",
    "1kq0y3i": "hot_take",
    "1kq0xju": "hot_take",
    "1kq0vrv": "hot_take",
    "1kq0u5x": "hot_take",
    "1kq0lyt": "meta_community",
    "1kq0cew": "meta_community",
    "1kq3dy6": "hot_take",
    "1kpzij9": "analysis",
    "1kpzlsu": "hot_take",
    "1kpp4ve": "news_updates",
    "1kpros2": "analysis",
    "1kplvup": "hot_take",
    "1kpm2ir": "meta_community",
    "1kps6ym": "meta_community",
    "1kpsfyb": "hot_take",
    "1kpwypt": "meta_community",
    "1kpww9w": "meta_community",
    "1kpy9ma": "meta_community",
    "1kpyclk": "meta_community",
    "1kpy24j": "meta_community",
    "1kpwl82": "meta_community",
    "1kpuug0m": "analysis",
    "1kpug0m": "analysis",
    "1kpxtst": "analysis",
    "1kpybat": "hot_take",
    "1kpygc7": "news_updates",
    "1kpyto8": "news_updates",
    "1kpy333": "news_updates",
    "1kpy3st": "news_updates",
    "1kpy9xv": "news_updates",
    "1kpypit": "hot_take",
    "1kpwroj": "hot_take",
    "1kpwu9y": "hot_take",
    "1kpw215": "meta_community",
    "1kpwlag": "hot_take",
    "1kpwjix": "analysis",
    "1kpwmb1": "hot_take",
    "1kpwk1b": "hot_take",
    "1kpyv6v": "hot_take",
    "1kpytsh": "hot_take",
    "1kpzc3k": "hot_take",
    "1kpz2ov": "hot_take",
    "1kpxgi8": "hot_take",
    "1kpxdmu": "hot_take",
    "1kpx3pw": "hot_take",
    "1kpw916": "hot_take",
    "1kpw71e": "hot_take",
    "1kpwqlx": "hot_take",
    "1kpvq2g": "news_updates",
    "1kpzx59": "analysis",
    "1kq0cjy": "analysis",
    "1kq0c8z": "analysis",
    "1kq0aji": "analysis",
    "1kpzmro": "analysis",
    "1kpzd3r": "analysis",
    "1kpxv55": "analysis",
    "1kpx4qv": "analysis",
    "1kpwrdl": "analysis",
    "1kpwg6r": "analysis",
    "1kpw9fp": "analysis",
    "1kpw47g": "analysis",
    "1kpw01h": "analysis",
    "1kpvxiq": "analysis",
    "1kpywjx": "analysis",
    "1kpy650": "analysis",
    "1kpyg6n": "analysis",
    "1kpw9n5": "analysis",
    "1kpw7tl": "analysis",
    "1kpw5hp": "analysis",
    "1kpw4ba": "analysis",
    "1kpw3yv": "analysis",
    "1kpw33a": "analysis",
    "1kpw1sc": "analysis",
    "1kpw0zj": "analysis",
    "1kpwqqi": "analysis",
    "1kpwqe9": "analysis",
    "1kpxhy6": "analysis",
    "1kpxe8t": "analysis",
    "1kpxf7u": "analysis",
    "1kpxcru": "analysis",
    "1kpxa6c": "analysis",
    "1kpx3e0": "analysis",
    "1kpx1g0": "analysis",
    "1kpz9eu": "analysis",
    "1kpz0zj": "analysis",
    "1kpwu7n": "meta_community",
    "1kpw2ld": "news_updates",
    "1kpwtay": "news_updates",
    "1kpws3w": "news_updates",
    "1kpwei9": "news_updates",
    "1kpwe88": "news_updates",
    "1kpwboa": "news_updates",
    "1kpwbmc": "news_updates",
    "1kpwa13": "news_updates",
    "1kpwpv2": "news_updates",
    "1kpxlup": "news_updates",
    "1kpxl6e": "news_updates",
    "1kpxij8": "news_updates",
    "1kpxgmb": "news_updates",
    "1kpxcu7": "news_updates",
    "1kpxb8u": "news_updates",
    "1kpx9kb": "news_updates",
    "1kpx3uf": "news_updates",
    "1kpw49f": "news_updates",
    "1kpw7ye": "analysis",
}


def extract_tag(text: str) -> str:
    m = re.match(r"^\[([^\]]+)\]", text.strip())
    return m.group(1).strip().lower() if m else ""


def word_count(text: str) -> int:
    return len(text.split())


def is_meta_thread(text: str) -> bool:
    lower = text.lower()
    if re.search(r"\[serious next day thread\]", lower):
        return True
    if re.search(r"^game thread:", lower):
        return True
    if "daily discussion thread" in lower and "game thread index" in lower:
        return True
    if "keep your memes" in lower or "direct replies to this post will be removed" in lower:
        return True
    return False


def is_meta_fandom(text: str) -> bool:
    lower = text.lower()
    patterns = [
        r"choose an nba team",
        r"from france",
        r"drift bottle",
        r"championship droughts across",
        r"thetvapp",
        r"doris burke",
        r"yankee stadium",
        r"caitlin clark",
        r"i have a confession",
        r"jokic fan with a confession",
        r"divisional rivalries",
        r"bill simmons and ryen",
        r"\[bill simmons\]",
        r"adam silver is doing a horrible job",
        r"commercials \(and using replay",
        r"devastated in a non narrative",
        r"what's the hype with new york knicks",
        r"why did the dallas cowboys trade",
        r"people with no ties with pacers",
        r"playoffs are .* this year",
    ]
    return any(re.search(p, lower) for p in patterns)


def is_news_relay(text: str, tag: str) -> bool:
    lower = text.lower()
    news_tags = (
        "charania", "woj", "shams", "highlight", "highlights", "lowlight",
        "all-access", "bobby marks", "stein", "scotto", "sam amick",
        "the athletic", "jeff stotts", "pelton", "murray", "uthayakumar",
        "benedetto", "post game thread", "nbatv",
    )
    if any(k in tag for k in news_tags):
        return True
    patterns = [
        r"postgame conference",
        r"postgame interview",
        r"full postgame",
        r"postgame presser",
        r"water leak during",
        r"available to play",
        r"will play and start",
        r"expected to warm up",
        r"underdognba",
        r"has been eliminated from championship contention",
        r"conference finals schedule:",
        r"when is nba mvp",
        r"when are the nba mvp",
        r"warm up an hour before tip-off",
        r"grade 2 strain",
        r"extension number will be at least",
        r"added .+ as an assistant coach",
        r"gives respect to the entire",
        r"gets a bit delayed due to some leaks",
        r"shoutout at the end of",
        r"on playing his cousin",
        r"on whether the nuggets can win",
        r"on what he'?ll do this offseason",
        r"on how this offseason",
        r"on playing through a hamstring",
        r"on the playoff scheduling",
        r"on what he'?s most proud",
        r"on if he felt nervous",
        r"after the game \"",
        r"after the nuggets were defeated",
        r"tried to get russell westbrook in the locker room",
    ]
    if any(re.search(p, lower) for p in patterns):
        return True
    # Short quote-only posts
    if re.search(r'^[^"]{0,80}:\s*"[^"]{10,}', text):
        return True
    return False


def is_structured_analysis(text: str) -> bool:
    lower = text.lower()
    patterns = [
        r"game 7 disappointments",
        r"net rating",
        r"\|rank\|team\|",
        r"repeat champions",
        r"playoff history:",
        r"remaining teams by salary",
        r"free throws attempted",
        r"collective bargaining",
        r"here's the math",
        r"per-game pay",
        r"series breakdown",
        r"point differential",
        r"gamescore",
        r"player efficiency rating",
        r"conference finals appearance",
        r"50-win teams",
        r"top 10 games by",
        r"for each of the 2019-2024 champs",
        r"for the first time in history",
        r"unique champions",
        r"different champions in \d+",
        r"margin of victory in a game 7",
        r"largest point differential",
        r"only \d+ players remaining",
        r"only \d+ of jokic",
        r"net points\" stat",
        r"how to win in the playoffs",
        r"title favorites according to espn bpi",
        r"knicks vs pistons and celtics in the regular season",
        r"travel(ed)? distance",
        r"stat padder",
        r"off season moves for",
        r"offseason moves for",
        r"went past 2nd round exactly",
        r"recorded a higher \\+/-",
        r"since 2023, rudy gobert",
        r"\d+/[\d.]+/\d+ on .+ splits.*\d+/[\d.]+/\d+ on",
        r"saw this floating around ig",
        r"100% field goal\?",
        r"single-game minimum and maximum team distances",
        r"western conference finals will be played outside",
        r"2025 west finals is the first",
        r"steals in this series",
        r"why the nba will never reduce the season",
    ]
    if any(re.search(p, lower) for p in patterns):
        return True
    if re.search(r"\|[^|]+\|[^|]+\|", text) and word_count(text) > 35:
        return True
    stat_lines = len(re.findall(r"\d+\.?\d*\s*(ppg|pts|fta|turnovers?|assists?|rebounds?)", lower))
    if stat_lines >= 4 and word_count(text) > 25:
        return True
    return False


def is_hot_take(text: str) -> bool:
    lower = text.lower()
    patterns = [
        r"\bgoat\b",
        r"face of the (nba|league)",
        r"who would you take",
        r"who ya got",
        r"who's your pick",
        r"unpopular opinion",
        r"hot take",
        r"overrated",
        r"washed",
        r"underachiever",
        r"should the .+ trade",
        r"better (passer|player)",
        r"rank them",
        r"does that make them",
        r"coldest moments?",
        r"legal now\?",
        r"is .+ crazy to say",
        r"come at me",
        r"liability",
        r"asking out",
        r"not a top 5",
        r"thoughts\?",
        r"what if",
        r"finals prediction when",
        r"likely matchups for",
        r"why is game 1 of the wcf",
        r"why the wcf starts before",
        r"what is going on with defense",
        r"hand checking legal",
        r"assist should be credited when",
        r"sign and trade need to be free agents",
        r"expansion franchises be in this era",
        r"predict the 2028",
        r"devin booker vs donovan mitchell",
        r"is there any theory on why",
        r"was this westbrook's last chance",
        r"whose individual performance are you",
        r"can jalen brunson be the",
        r"epic battle for the title",
        r"down to the final four",
    ]
    return any(re.search(p, lower) for p in patterns)


def label_post(post_id: str, text: str) -> tuple[str, str]:
    if post_id in OVERRIDES:
        return OVERRIDES[post_id], "override"

    tag = extract_tag(text)

    if is_meta_thread(text):
        return "meta_community", "thread"
    if is_news_relay(text, tag):
        return "news_updates", "news"
    if is_meta_fandom(text):
        return "meta_community", "meta"
    if is_structured_analysis(text):
        return "analysis", "analysis"
    if is_hot_take(text):
        return "hot_take", "hot_take"

    if "?" in text and word_count(text) < 150:
        return "hot_take", "question"
    if re.search(r"\d+\.?\d*\s*(ppg|pts|%|fta)", text.lower()):
        return "analysis", "stats"
    return "hot_take", "fallback"


def main() -> None:
    root = Path(__file__).parent
    src = root / "data" / "r_nba_to_annotate.csv"
    df = pd.read_csv(src)

    labels, notes = [], []
    for _, row in df.iterrows():
        label, reason = label_post(str(row["id"]), str(row["text"]))
        assert label in VALID_LABELS
        labels.append(label)
        notes.append(reason)

    df["label"] = labels
    df["notes"] = notes

    print(f"Distribution (all {len(df)} rows):")
    print(df["label"].value_counts().sort_index())
    pct = df["label"].value_counts(normalize=True).max()
    print(f"Max class share: {pct:.1%}\n")

    df.to_csv(src, index=False)
    labeled = df[["text", "label"]].copy()
    out = root / "data" / "r_nba_labeled.csv"
    labeled.to_csv(out, index=False)
    print(f"Saved {len(labeled)} rows to {out}")


if __name__ == "__main__":
    main()
