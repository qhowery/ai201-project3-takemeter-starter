#!/usr/bin/env python3
"""Collect public r/nba submissions for TakeMeter annotation.

Outputs CSV with columns: id, text, label, notes
- label starts empty — you must annotate manually (or pre-label + review)
- notes can record source hints or difficult-case reasoning

Usage:
  python collect_r_nba_posts.py
  python collect_r_nba_posts.py --output data/r_nba_to_annotate.csv --target 260
"""

from __future__ import annotations

import argparse
import csv
import re
import time
import json
import ssl
import subprocess
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


API = "https://api.pullpush.io/reddit/search/submission/"
MIN_TEXT_LEN = 40
MAX_TEXT_LEN = 2000


def fetch_batch(subreddit: str, size: int = 100, before: int | None = None) -> list[dict]:
    params = {
        "subreddit": subreddit,
        "size": size,
        "sort": "desc",
        "sort_type": "created_utc",
    }
    if before:
        params["before"] = before
    url = API + "?" + urlencode(params)
    try:
        ctx = ssl.create_default_context()
        with urlopen(url, timeout=30, context=ctx) as resp:
            data = json.load(resp)
    except ssl.SSLError:
        raw = subprocess.check_output(["curl", "-s", url], timeout=60)
        data = json.loads(raw)
    return data.get("data", [])


def build_text(title: str, selftext: str) -> str:
    title = (title or "").strip()
    body = (selftext or "").strip()
    if body and body.lower() not in ("[removed]", "[deleted]"):
        text = f"{title}\n\n{body}" if title else body
    else:
        text = title
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_TEXT_LEN]


def is_usable(post: dict) -> bool:
    if post.get("removed_by_category"):
        return False
    text = build_text(post.get("title", ""), post.get("selftext", ""))
    if len(text) < MIN_TEXT_LEN:
        return False
    if text.lower() in ("[removed]", "[deleted]"):
        return False
    return True


def collect(subreddit: str, target: int) -> list[dict]:
    seen_ids: set[str] = set()
    rows: list[dict] = []
    before: int | None = None
    attempts = 0

    while len(rows) < target and attempts < 30:
        attempts += 1
        batch = fetch_batch(subreddit, size=100, before=before)
        if not batch:
            break
        before = batch[-1].get("created_utc")
        for post in batch:
            pid = post.get("id")
            if not pid or pid in seen_ids:
                continue
            if not is_usable(post):
                continue
            seen_ids.add(pid)
            text = build_text(post.get("title", ""), post.get("selftext", ""))
            rows.append(
                {
                    "id": pid,
                    "text": text,
                    "label": "",
                    "notes": f"score={post.get('score', 0)}",
                }
            )
            if len(rows) >= target:
                break
        print(f"Collected {len(rows)}/{target}...")
        time.sleep(0.5)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subreddit", default="nba")
    parser.add_argument("--target", type=int, default=260)
    parser.add_argument("--output", default="data/r_nba_to_annotate.csv")
    args = parser.parse_args()

    rows = collect(args.subreddit, args.target)
    if len(rows) < 200:
        raise SystemExit(f"Only collected {len(rows)} posts — need at least 200.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "text", "label", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} posts to {args.output}")
    print("Next: fill the 'label' column using planning.md definitions, then save as r_nba_labeled.csv")


if __name__ == "__main__":
    main()
