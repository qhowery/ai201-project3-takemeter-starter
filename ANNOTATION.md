# Milestone 3 — Annotation Guide

## Files

| File | Purpose |
|------|---------|
| `data/r_nba_to_annotate.csv` | Collected posts — fill in `label` column |
| `data/r_nba_labeled.csv` | Final file for Colab (≥200 rows, `text` + `label`) |
| `planning.md` | Label definitions and tie-breakers |

## Label values (exact strings)

```
analysis
hot_take
news_updates
meta_community
```

## Workflow

1. Open `data/r_nba_to_annotate.csv` in Google Sheets or Excel.
2. Read each `text` row; assign one `label` using `planning.md`.
3. For hard cases, write your reasoning in `notes`.
4. Log at least **3 hard cases** in `planning.md` Section 3 annotation table.
5. Count labels — **no single label > 70%** of your final 200+.
6. Save as **`data/r_nba_labeled.csv`** with columns `text`, `label`.

## Re-collect if needed

```bash
python3 collect_r_nba_posts.py --target 260 --output data/r_nba_to_annotate.csv
```

## After labeling

```bash
python3 label_r_nba_posts.py
```

Upload `data/r_nba_labeled.csv` in Colab notebook Section 1 → run Milestones 4–6.
