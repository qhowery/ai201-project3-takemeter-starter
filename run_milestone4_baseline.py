#!/usr/bin/env python3
"""Milestone 4 — Groq zero-shot baseline on locked 15% test split."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
from groq import Groq
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "data" / "r_nba_labeled.csv"
OUT_PATH = ROOT / "baseline_evaluation.json"

LABEL_MAP = {
    "analysis": 0,
    "hot_take": 1,
    "news_updates": 2,
    "meta_community": 3,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

SYSTEM_PROMPT = """
You are classifying Reddit posts from r/nba.
Assign each post to exactly one category based on its PRIMARY purpose.

analysis: Makes a basketball argument supported by stats, film, matchup logic, historical context, or step-by-step reasoning.
Example: "Game 7 Disappointments: Over the last 12 game 7s, 10 of them have been decided by double digits. Half have been decided by 20+."

hot_take: Asserts a bold opinion about players, teams, or outcomes without meaningful evidence or structured reasoning.
Example: "Can Jalen Brunson be the face of the NBA if the Knicks win a championship? LeBron and Steph are in the twilight zone anyway."

news_updates: Primarily relays factual information (injuries, trades, scores, schedules, official announcements) with little argumentative framing.
Example: "[Charania] Nets star Kyrie Irving has entered COVID-19 health and safety protocols."

meta_community: Discusses fandom, media, the subreddit itself, or off-court culture rather than evaluating on-court performance.
Example: "[SERIOUS NEXT DAY THREAD] Post-Game Discussion — keep memes and reactions in the other thread."

Tie-breaker: If one cherry-picked stat supports an extreme conclusion without context, label hot_take (not analysis).

Respond with ONLY one label name from the list below. No explanation.

Valid labels:
analysis
hot_take
news_updates
meta_community
"""


def classify_with_groq(client: Groq, text: str) -> str | None:
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify this post:\n\n{text}"},
            ],
            temperature=0,
            max_tokens=20,
        )
        raw = response.choices[0].message.content.strip().lower()
        for label in sorted(LABEL_MAP, key=len, reverse=True):
            if raw == label or label in raw:
                return label
        return None
    except Exception as exc:
        print(f"API error: {exc}")
        return None


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY in your environment before running.")

    df = pd.read_csv(CSV_PATH)
    df["label_id"] = df["label"].map(LABEL_MAP)
    df = df.dropna(subset=["label_id"])
    df["label_id"] = df["label_id"].astype(int)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["label_id"]
    )
    _, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label_id"]
    )
    test_df = test_df.reset_index(drop=True)

    client = Groq(api_key=api_key)
    print(f"Running Groq baseline on {len(test_df)} test examples...")

    baseline_preds: list[str | None] = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        pred = classify_with_groq(client, row["text"])
        baseline_preds.append(pred)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(test_df)} complete...")
        time.sleep(0.1)

    none_count = baseline_preds.count(None)
    valid = [(p, t) for p, t in zip(baseline_preds, test_df["label_id"]) if p is not None]
    bl_pred_ids = [LABEL_MAP[p] for p, _ in valid]
    bl_true_ids = [t for _, t in valid]
    label_names = [ID_TO_LABEL[i] for i in range(len(LABEL_MAP))]

    bl_accuracy = accuracy_score(bl_true_ids, bl_pred_ids)
    report = classification_report(
        bl_true_ids, bl_pred_ids, target_names=label_names, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(bl_true_ids, bl_pred_ids).tolist()

    print(f"\nBaseline accuracy: {bl_accuracy:.3f}")
    print(f"Parseable: {len(valid)}/{len(test_df)} ({none_count} unparseable)")
    print("\nPer-class metrics:")
    print(classification_report(bl_true_ids, bl_pred_ids, target_names=label_names, zero_division=0))

    results = {
        "milestone": 4,
        "model": "llama-3.3-70b-versatile",
        "baseline_accuracy": round(bl_accuracy, 4),
        "test_set_size": len(test_df),
        "parseable_count": len(valid),
        "unparseable_count": none_count,
        "unparseable_rate": round(none_count / len(test_df), 4),
        "label_map": LABEL_MAP,
        "classification_report": report,
        "confusion_matrix": cm,
        "confusion_matrix_labels": label_names,
        "split": {"train": len(train_df), "test": len(test_df), "random_state": 42},
    }

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
