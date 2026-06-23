# TakeMeter — AI201 Project 3

**Student:** Quonlee Howery · qhowery@princeton.edu  
**Community:** r/nba · **Labels:** analysis, hot_take, news_updates, meta_community

## What this project does

TakeMeter classifies r/nba posts by discourse type so a feed could surface thoughtful **analysis** and deprioritize low-effort **hot takes** during busy events (playoffs, trade deadline).

## Project structure

```
ai201-project3-takemeter-starter/
├── planning.md                          # Milestones 1–2 spec + annotation log
├── ai201_project3_takemeter_starter_clean.ipynb
├── data/
│   ├── r_nba_labeled.csv                # 300 labeled posts (training)
│   └── r_nba_to_annotate.csv            # raw collection + label notes
├── collect_r_nba_posts.py               # PullPush data collection
├── label_r_nba_posts.py                 # Rule-based labeling helper
├── run_milestone4_baseline.py           # Groq zero-shot baseline
├── run_milestone5_finetune.py           # DistilBERT fine-tuning
├── evaluation_results.json              # Final metrics
├── baseline_evaluation.json             # Baseline-only metrics
├── confusion_matrix.png                 # Test-set confusion matrix
├── demo_notes.txt                       # Video script (Milestone 6)
├── ANNOTATION.md                        # Labeling workflow
└── requirements.txt
```

## Results (test set = 45 posts, 15% holdout)

| Model | Accuracy |
|-------|----------|
| Zero-shot Groq (`llama-3.3-70b-versatile`) | **0.733** |
| Fine-tuned DistilBERT | **0.822** |
| **Improvement** | **+0.089** |

### Per-class F1 (fine-tuned)

| Label | F1 |
|-------|-----|
| analysis | 0.63 |
| hot_take | 0.73 |
| news_updates | 0.97 |
| meta_community | 0.90 |

**Success criteria met:** beat baseline by ≥ +0.08, all per-class F1 ≥ 0.60, no class > 65% of predictions.

First training run underperformed (0.711) because the model ignored `hot_take`. Retrained with **class-weighted loss** and **5 epochs** — fixed minority-class collapse.

## Top confused pair

**`analysis` ↔ `hot_take`** — posts with a stat attached to a bold ranking. See `planning.md` Section 3.

## How to reproduce

```bash
cd ai201-project3-takemeter-starter
pip install -r requirements.txt
cp .env.example .env   # add GROQ_API_KEY

export GROQ_API_KEY='your-key'
python3 run_milestone4_baseline.py
python3 run_milestone5_finetune.py
```

Colab: open the notebook, upload `data/r_nba_labeled.csv`, set `GROQ_API_KEY` in Secrets, run all cells.

## Demo video

**[Watch demo on Google Drive](https://drive.google.com/file/d/1eL2u_rMc2IyTLbaFcl9Xtjt6Z1n2Va2K/view?usp=share_link)**

See **`demo_notes.txt`** for the ~3-minute script.

## AI tool usage

- Cursor / Claude: planning, labeling scripts, training scripts, README, demo script
- Groq: zero-shot baseline only (labels manually reviewed per `planning.md`)

## Submission checklist

- [x] `planning.md`
- [x] `data/r_nba_labeled.csv`
- [x] `evaluation_results.json`
- [x] `confusion_matrix.png`
- [x] `README.md`
- [x] [Demo video](https://drive.google.com/file/d/1eL2u_rMc2IyTLbaFcl9Xtjt6Z1n2Va2K/view?usp=share_link)
