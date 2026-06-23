# TakeMeter — planning.md

**Student:** Quonlee Howery · qhowery@princeton.edu  
**Community:** r/nba · **Labels:** 4 · **Target dataset:** 200+ annotated posts

---

## 1. Community

**What community did you choose and why?**

I chose **r/nba** (https://www.reddit.com/r/nba/), Reddit's main NBA discussion hub. It is active year-round, text-heavy (titles, self-posts, game threads, rumor posts), and publicly accessible at scale.

**Why is it a good fit for classification?**

Discourse quality varies widely in the same feed: Woj/Shams injury tweets sit next to stat breakdowns, meme reactions, and unsupported "X is washed" posts. Regulars implicitly rank **analysis** above **hot takes** — upvotes and comment tone reflect that. That makes the label boundaries meaningful rather than arbitrary: I'm classifying types of contribution the community already distinguishes, not inventing categories from thin air.

**Data unit:** One public Reddit **submission** (post title + self-text combined into a single `text` field). Minimum 40 characters after cleaning; max 2000 characters truncated for training stability.

---

## 2. Labels

Each post gets **one label** based on **primary purpose** (not tone, popularity, or comment count).

CSV label strings (must match notebook `LABEL_MAP` exactly):

| CSV label | Display name | Definition |
|-----------|--------------|------------|
| `analysis` | Analysis | Makes a basketball argument supported by stats, film, matchup logic, historical context, or step-by-step reasoning. |
| `hot_take` | Hot Take | Asserts a bold opinion about players, teams, or outcomes without meaningful evidence or structured reasoning. |
| `news_updates` | News & Updates | Primarily relays factual information (injuries, trades, scores, schedules, official announcements) with little argumentative framing. |
| `meta_community` | Meta / Community | Discusses fandom, media, the subreddit itself, or off-court culture rather than evaluating on-court performance. |

### Examples per label

**analysis**
1. *"Game 7 Disappointments: Over the last 12 game 7s, 10 of them have been decided by double digits..."* — structured historical stats.
2. *"I analyzed Jaren Jackson Jr.'s home vs. away block stats; the scorekeeper discrepancy suggests inflated defensive numbers."* — investigative comparison.

**hot_take**
1. *"Can Jalen Brunson be the face of the NBA if the Knicks win a championship?"* — speculative ranking, no evidence.
2. *"Max Kellerman on LeBron: it's called selling out when there's no cost to your stand."* — pundit opinion relay without argument.

**news_updates**
1. *"[Charania] Nets star Kyrie Irving has entered COVID-19 health and safety protocols."*
2. *"[Highlight] Luka Doncic hits the game-winning three at the buzzer."*

**meta_community**
1. *"[SERIOUS NEXT DAY THREAD] Post-Game Discussion — keep memes in the other thread."*
2. *"Championship droughts across NBA/NHL/NFL/MLB for NYC, Indy, Minneapolis, OKC"* — civic fandom comparison.

**Mutual exclusivity:** Ask: Is the main move (A) unsupported opinion, (B) evidence-backed argument, (C) fact relay, or (D) community/media commentary? Most posts map to one bucket using tie-breakers below.

---

## 3. Hard Edge Cases

### Primary ambiguous type: Hot Take ↔ Analysis

**Pattern:** One cherry-picked stat supporting a sweeping ranking or legacy claim.

| Example | Decision | Reason |
|---------|----------|--------|
| *"Tatum's 34% on wide-open threes proves he can't be a #1 option on a champion."* | `hot_take` | Stat present but conclusion outruns evidence; no context. |
| Same stat + discussion of defensive schemes, shot quality, sample size | `analysis` | Main move is explaining how evidence supports a narrower claim. |

**Handling rule (primary-move):** Label `analysis` only when the post *explains how* evidence supports a proportionate conclusion. Otherwise `hot_take`, even if a number appears.

### Secondary ambiguous types

| Pattern | Labels in tension | Rule |
|---------|-------------------|------|
| *"Shams: Team exploring trade for Player X — thoughts?"* | news vs hot_take/analysis | Mostly report + brief prompt → `news_updates`; body is trade logic → `analysis` or `hot_take`. |
| *"ESPN hypes the Lakers even when .500"* | meta vs hot_take | Media criticism → `meta_community`; "Lakers aren't good" judgment → `hot_take`. |
| Vague frustration about "playoff defense" with no specifics | hot_take vs analysis | No cited games/metrics → `hot_take`. |

### Annotation log (fill as you label — Milestone 3)

| Post snippet (paraphrase) | Could be | Decided | Why |
|---------------------------|----------|---------|-----|
| *"Is it generally agreed Jokic passes too much?"* (mentions playoff TOs) | `analysis` / `hot_take` | `hot_take` | Stats appear but post doesn't explain *how* evidence supports a narrow claim — primary move is provocative opinion (Section 3 primary-move rule). |
| *"[Murray] Thunder had 74 steals in this series…"* | `news_updates` / `analysis` | `analysis` | Reporter tag looks like news, but body is a structured historical stat comparison with context — main move is evidence, not fact relay. |
| *"Nikola Jokic 2025 Playoffs stats — 26.2 PPG on 48.9/38.0/77.2…"* | `news_updates` / `analysis` | `news_updates` | Pure stat-line listing with no argumentative framing — factual relay even though numbers are present. |
| *"Championship droughts across NBA/NHL/NFL/MLB for NYC, Indy, OKC"* | `analysis` / `meta_community` | `meta_community` | Table is civic fandom comparison across sports, not an on-court basketball argument (planning.md example). |
| *"2015 Warriors vs 2025 Thunder — who would you take?"* ( cites ORtg/DRtg ) | `analysis` / `hot_take` | `hot_take` | Cherry-picked team-level stats supporting a hypothetical matchup vote — conclusion outruns evidence. |

---

## 4. Data Collection Plan

**Source:** Public r/nba submissions via [PullPush archive API](https://api.pullpush.io/reddit/search/submission/) (`collect_r_nba_posts.py` in repo root). Manual review only — no private content.

**Target size:** **260 posts collected** → annotate **≥200** for training (buffer for dropped/ambiguous rows).

**Per-label targets (no label > 70% of final set):**

| Label | Target count | Max allowed (70%) |
|-------|-------------|-------------------|
| `analysis` | 55 | 140 |
| `hot_take` | 55 | 140 |
| `news_updates` | 55 | 140 |
| `meta_community` | 55 | 140 |

**Collection mix:** Recent submissions across multiple fetch batches (varying scores/dates) so the set isn't all one event type (e.g., only trade deadline).

**If imbalanced after 200 labels:** Stop and collect additional posts matching underrepresented labels (e.g., search for `[Charania]`, `[Highlight]`, `SERIOUS`, opinion titles) before training.

**Output files:**
- `data/r_nba_to_annotate.csv` — raw collection (`id`, `text`, `label`, `notes`)
- `data/r_nba_labeled.csv` — final annotated file for Colab notebook (`text`, `label` required; `notes` optional)

**Annotation workflow:**
1. Read each post fully; assign one label from definitions above.
2. Record hard cases in `notes` column and annotation log (Section 3).
3. Count per-label totals; rebalance if any label > 70%.
4. Do **not** bulk-skim — inconsistent labels hurt the model more than small dataset size.

**Pre-labeling (optional):** May use Groq/Claude with label definitions to suggest labels on unlabeled batch; **every row manually reviewed and corrected**. Track pre-labeled rows in `notes` (e.g., `pre-labeled: groq, reviewed: yes`).

---

## 5. Evaluation Metrics

**Accuracy alone is insufficient** because class imbalance would let a model predict the majority label and look "good."

| Metric | Why it matters for this task |
|--------|------------------------------|
| **Overall accuracy** | Headline comparison: baseline vs fine-tuned on locked 15% test set. |
| **Per-class precision** | Of posts predicted as `analysis`, how many truly are? High precision + low recall = conservative classifier. |
| **Per-class recall** | Of true `hot_take` posts, how many did we catch? Low recall on one class = missing that discourse type. |
| **Per-class F1** | Single balanced score per label; primary success metric per class. |
| **Confusion matrix** | Shows *direction* of errors (e.g., `analysis` → `hot_take`), which maps directly to label-boundary problems. |

**Baseline comparison:** Zero-shot Groq on same test split — fine-tuning should beat baseline on overall accuracy **and** on F1 for at least 3 of 4 classes.

**What I will not optimize for:** Training accuracy alone (overfitting risk on 200 examples).

---

## 6. Definition of Success

**Deployment context (hypothetical):** A feed-ranking tool that surfaces `analysis` and deprioritizes low-effort `hot_take` during high-traffic events (playoffs, trade deadline).

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Beat baseline | Fine-tuned accuracy ≥ baseline + **0.08** on test set | Fine-tuning must add real value over zero-shot. |
| Per-class F1 | **≥ 0.60** for each of 4 labels | Usable per-class ranking; one class at F1≈0 is not deployable. |
| No majority collapse | No single label > **65%** of predictions on test set | Model isn't defaulting to one class. |
| Parseable baseline | Groq baseline unparseable rate **< 10%** | Prompt is well-formed enough to compare fairly. |
| Actionable errors | Can name **top confused pair** from confusion matrix + 3 analyzed failures | Understand *why* the model fails, not just accuracy %. |

**"Good enough" for this course project:** Meet all rows above **or** document clearly why not (e.g., label inconsistency discovered, specific class needs more data) with a concrete fix plan.

**Stretch success:** All per-class F1 ≥ **0.70** and fine-tuned beats baseline by ≥ **0.12**.

---

## 7. AI Tool Plan

### Label stress-testing (before annotation)

**Tool:** Claude / Cursor  
**Input:** Label definitions + primary edge case (Section 3)  
**Prompt:** *"Generate 10 r/nba-style posts that sit on the boundary between hot_take and analysis."*  
**Use:** If I can't classify ≥2 posts cleanly, tighten definitions before annotating 200 examples.  
**Decision:** Run once before bulk annotation; revise tie-breaker if needed.

### Annotation assistance (optional)

**Tool:** Groq (`llama-3.3-70b-versatile`)  
**Workflow:** Batch of 20–30 unlabeled posts + label definitions → model suggests labels → I review every row and correct.  
**Tracking:** `notes` column: `pre-labeled: groq, reviewed: yes, changed: hot_take→analysis`  
**Decision:** Use for **speed only**, not replacement for reading posts. Disclose in README AI usage section.

### Failure analysis (after fine-tuning)

**Tool:** Claude / Cursor  
**Input:** List of wrong predictions from notebook (text, true label, predicted label, confidence)  
**Prompt:** *"Identify patterns in these misclassifications. Which label pairs confuse most? Short posts? Sarcasm?"*  
**Verification:** Re-read flagged examples myself; discard AI patterns that don't hold on manual review.  
**Output:** Feeds README failure analysis (≥3 specific wrong examples with root cause).

---

## Milestone Checklist

- [x] Milestone 1 — Community + labels defined
- [x] Milestone 2 — planning.md six questions + AI Tool Plan
- [x] Milestone 3 — 200+ labeled rows in `r_nba_labeled.csv`, balanced, 3+ hard cases logged
- [x] Milestone 4 — Baseline Groq metrics on test set (accuracy **0.733**, 45/45 parseable, see `baseline_evaluation.json`)
- [x] Milestone 5 — Fine-tuned DistilBERT + comparison (accuracy **0.822**, +0.089 vs baseline, see `evaluation_results.json`)
- [ ] Milestone 6 — README evaluation report + demo video (script in `demo_notes.txt`)
