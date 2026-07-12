# CRISP-DM — Steps 1 & 2: Business & Data Understanding
## FlyRank Content Refresh Prioritisation Model

---

## Step 1 — Business Understanding

### 1.1 The Business Problem

FlyRank is a content-as-infrastructure company: it researches, writes, publishes, and optimises
content inside client websites — using algorithms rather than people doing it by hand.

The product already "works" today in the sense that it has **hand-written rule flags** that
assign each page a health score and action tags (`needs-ctr-fix`, `is-quick-win`, etc.).
These rules are transparent and run in production. The problem is that they run out of power as the
signals multiply, intertwine, and shift over time — no trained model has replaced them yet.

**The specific pain:** published content quietly decays. Rankings slip, clicks drop, and most teams
notice too late. Out of thousands of pages per client, someone must decide which one to fix *first*.
That decision is currently driven by hand-written thresholds. The hypothesis this project tests is:
a learned model can beat those thresholds.

---

### 1.2 The Four Framing Questions (from `skills/framing-ml-problems/SKILL.md`)

| # | Question | Answer |
|---|---|---|
| 1 | **What decision does this improve?** | Which page should a content editor refresh first? — a ranking / prioritisation decision, made daily, across thousands of pages. |
| 2 | **Who acts on the output, and what do they do?** | A FlyRank content team member (or an automated workflow) receives the ranked refresh queue and opens the top-N pages for review and update. Without a better ranking, effort is wasted on stable pages while declining ones are missed. |
| 3 | **What does a wrong answer cost?** | A **false negative** (missing a declining page) costs client rankings — pages that should have been fixed decay further. A **false positive** (sending an editor to a healthy page) wastes editing time. Both matter, so **Precision@K** (how many of the top K flagged pages are truly declining) is the primary metric — it is the business-meaningful error rate on editor hours. |
| 4 | **Why does ML beat a plain rule?** | The decision correlates with many signals simultaneously: impressions trend, position, CTR, content age, engagement, scroll, AI traffic, keyword competition — all interacting. Hand-thresholds can capture one or two; a model captures the joint pattern, including interactions. The reference pipeline proves it: `Precision@50 = 0.240` (hand-rule baseline) → `0.680` (random forest) — **~2.8× lift**. |

---

### 1.3 ML Task Type Mapping

Using the task-type table from `framing-ml-problems`:

- The question is **"which ones first?"** → **Ranking / Scoring task**
- The underlying signal we rank on is **"is this page declining?"** → **Binary Classification** as the inner model
- The composite output is a **priority score** (blending model probability + rule-based reason codes)
- Primary metric: **Precision@50** (business-facing) | Secondary: **ROC-AUC**, **Average Precision**

---

### 1.4 The One-Paragraph Frame

> For FlyRank content editors deciding **which page to refresh first**, we will build a
> **ranked refresh queue** from 44 anonymised search-performance signals (Google Search Console +
> Google Analytics), predicting `is_declining_label = (trend_direction == "down")`,
> evaluated by **Precision@50**. A wrong call costs wasted editor hours (false positive) or
> a missed ranking decline (false negative). A plain threshold rule isn't enough because the
> decline signal is correlated with many shifting signals simultaneously — position tier, CTR,
> impressions trajectory, content age — not any single one. We will claim only **observed,
> directional, decision-support** results.

---

### 1.5 Critical Leakage Warning

> [!CAUTION]
> **`trend_direction` and `trend_pct` are the label source and are NEVER model features.**
> They encode the decision already made. Using them would make the model learn the label,
> not the real world.
>
> **Product flags (`health_score`, `needs_ctr_fix`, `is_quick_win`, …) are outputs, not inputs.**
> They are the baseline to beat. Using them as features is circular.

---

### 1.6 Success Criteria

| Criterion | Target | Rationale |
|---|---|---|
| Precision@50 ≥ 0.50 | Beat baseline by >2× | The hand-rule achieves 0.240; 2× is a meaningful lift |
| ROC-AUC > 0.70 | Better than chance + baseline | 0.627 (baseline), 0.747 (random forest reference) |
| No data leakage | `trend_direction` / `trend_pct` absent from features | Verified by leakage notebook (ML-05) |
| Client-holdout split | No client appears in both train and test | Prevents inter-client data bleeding |
| Claim language | Observed / measured / directional / decision-support | FlyRank house culture |

---

## Step 2 — Data Understanding

### 2.1 Dataset Overview

| Property | Value |
|---|---|
| File | `data/raw/content_refresh_anonymized.csv` |
| Rows | 30,000 |
| Columns | 44 |
| Grain | One row = one pseudonymised content item (page), one client, trailing-90-day window |
| Clients | 32 pseudonymised clients |
| Content types | keyword article (27,207), feedly article (2,096), comparison article (697) |
| Label | `is_declining_label = (trend_direction == "down")` |

---

### 2.2 Label Distribution

| Class | Count | Share |
|---|---|---|
| `is_declining_label = True` (down) | 16,262 | **54.2%** |
| `is_declining_label = False` | 13,738 | 45.8% |

> [!NOTE]
> The dataset is **mildly imbalanced toward declining pages** (54/46). This reflects the
> real product population but means a naïve "always decline" classifier would achieve 54%
> accuracy — always report class share before training. Precision@50 is immune to this
> because it evaluates the top-K ranked items, not the full dataset.

**`trend_direction` full breakdown:**

| Direction | Count | Meaning |
|---|---|---|
| `down` | 16,262 | Impressions fell > 20% last-30d vs prev-30d → **the label** |
| `stable` | 5,962 | < ±20% change |
| `up` | 4,388 | Rose > 20% |
| `new` | 2,236 | Previous window was 0, now > 0 (newly indexed) |
| `flat` | 1,152 | Both windows are 0 |

---

### 2.3 Column Groups & Key Gotchas

#### Identifiers (never features)
- `content_id` — pseudonymous page ID; unique per row; grouping/joining only
- `client_id` — 32 distinct values; **use for client-holdout train/test splits**

#### Keyword context (from content metadata)
- `search_volume`, `competition`, `cpc`, `competition_level`, `main_intent` — **8.2–8.7% missing**, all concentrated in `feedly article` rows (2,096 / 2,096 missing from feedly). A blind `fillna(0)` secretly encodes `content_type` into the feature → **add `has_keyword_data` flag instead**.

#### Content properties
- `word_count` / `char_count` — **25.7% missing** (7,699 rows). Missingness is NOT random — concentrated in one content type. Same solution: `has_word_count` flag.
- `provider_used` — 71.5% missing; not recommended as a feature.
- `model_used` — 19.1% missing; not recommended as a feature.

#### 90-day activity totals (core signal group)
- `impressions_90d`, `clicks_90d`, `sessions_90d`, `users_90d`, `engaged_sessions_90d`, `ai_sessions_90d`, `scroll_events_90d`, `days_with_impressions`, `days_with_sessions`
- Distributions are **heavily right-skewed** (median impressions = 731, mean = 5,200, max = 517,715) → log-transform before modeling (the reference pipeline already does `log_impressions_90d`).

#### 30-day comparison windows (label source — handle carefully)
- `impressions_last_30d`, `clicks_last_30d`, `sessions_last_30d`, `impressions_prev_30d`, `clicks_prev_30d`, `sessions_prev_30d`
- These **feed directly into `trend_pct` → `trend_direction` → the label**. Using them raw as features leaks the label. They can appear as features only if carefully lagged / pre-period-bounded (advanced work).

#### Derived rates (all ×100 percentages — not 0–1!)
| Column | Formula | Gotcha |
|---|---|---|
| `ctr` | `clicks_90d / impressions_90d × 100` | `ctr = 0.76` means **0.76%**, not 76% |
| `avg_position` | Mean GSC position | **`avg_position = 0` → "no data"**, not rank 0. Affects 1,205 rows |
| `engagement_rate` | `engaged_sessions / sessions × 100` | Normal 0–100 |
| `scroll_rate` | `scroll_events / pageviews × 100` | **Can exceed 100** (multiple scrolls per pageview). 119 rows > 100 |
| `ai_traffic_pct` | `ai_sessions / sessions × 100` | **Can exceed 100** (independent measurement systems). 23 rows > 100 |
| `trend_pct` | `(last_30d − prev_30d) / prev_30d × 100` | **Label source — NEVER a feature**. 3,388 missing (prev_30d = 0) |

---

### 2.4 Missingness Summary

| Column | Missing | % | Root Cause |
|---|---|---|---|
| `provider_used` | 21,438 | 71.5% | Not recorded at content creation |
| `word_count` / `char_count` / tiers | 7,699 | 25.7% | Not measured for some content types |
| `model_used` | 5,733 | 19.1% | Not recorded |
| `trend_pct` | 3,388 | 11.3% | `impressions_prev_30d = 0` (new content) |
| `search_volume` / `competition` / `cpc` | 2,468 | 8.2% | `feedly article` has no keyword data |
| `main_intent` / `competition_level` | 2,374–2,610 | 7.9–8.7% | Same root cause |
| `scroll_rate` | 125 | 0.4% | `pageviews_90d = 0` |

> [!WARNING]
> Missingness is **systematic by `content_type`**, not random. A `fillna(0)` on keyword columns
> silently encodes `feedly article` into the feature space. The safe pattern is:
> add `has_keyword_data = search_volume.notna().astype(int)` and then impute with median.

---

### 2.5 Key Signal Distributions

| Signal | Min | Median | Mean | Max | Note |
|---|---|---|---|---|---|
| `impressions_90d` | 1 | 731 | 5,200 | 517,715 | Log-transform essential |
| `clicks_90d` | 0 | 1 | 16 | 4,178 | Log-transform |
| `sessions_90d` | 1 | 7 | 37 | 4,345 | Log-transform |
| `ctr` | 0 | 0.07 | 0.51 | 100 | ×100 percentage |
| `avg_position` | 0 | 10.8 | 16.3 | 245 | 0 = no data |
| `engagement_rate` | 0 | 0 | 2.5 | 100 | Sparse — many zeros |
| `content_age_days` | 90 | 236 | 256 | 564 | All ≥ 90 in this slice |
| `word_count` | 8 | 2,877 | 3,108 | 9,546 | 25.7% missing |
| `search_volume` | 0 | 10 | 159 | 74,000 | Highly right-skewed |

---

### 2.6 Initial Signal–Label Relationships

Computed from the 30k dataset (medians by `trend_direction`):

| Signal | `down` median | `up` median | Interpretation |
|---|---|---|---|
| `impressions_90d` | 961 | 587 | Declining pages have **more** impressions (more visible = more measurable decay) |
| `avg_position` | 11.3 | 15.3 | Declining pages rank **better** — higher-rank pages are the ones being watched closely |
| `content_age_days` | 216 | 292 | Declining pages are **younger** — older content that has recovered is stable/up |
| `word_count` | 2,909 | 2,848 | Near-identical — **word count is not a lever** (Discovery C) |
| `ctr` | 0.10 | 0.10 | CTR alone doesn't separate declining from growing pages |

> [!NOTE]
> **Counter-intuitive finding:** Declining pages tend to have *higher* impressions and *better*
> positions than growing ones. This makes sense: pages that are visible enough to be tracked
> are the ones showing measurable decline. Low-impression "new" pages don't show up as "down"
> — they show up as "new". This means simple threshold rules on impressions alone will misflag.

---

### 2.7 Position Tier vs Trend Direction (Crosstab)

| Position tier | down | flat | new | stable | up |
|---|---|---|---|---|---|
| `top_3` | **24.1%** | 8.9% | **57.0%** | 7.2% | 2.8% |
| `striking` (≤20) | **61.0%** | 1.6% | 2.7% | 21.4% | 13.3% |
| `page_1` (≤10) | **57.0%** | 5.9% | 3.2% | 21.1% | 12.7% |
| `page_3_5` (≤50) | **56.2%** | 1.0% | 2.9% | 21.3% | 18.7% |
| `deep` (>50) | 34.4% | 4.1% | 9.8% | 14.5% | **37.2%** |

> [!IMPORTANT]
> The **`striking` and `page_1` tiers carry the highest decline rate (~57–61%)**.
> Pages that rank in the search sweet spot (positions 3–20) are the most at risk of tracked
> decline — they were once visible, and now they're slipping. Deep pages (>50) are more likely
> growing (new content climbing). This is a strong signal for the ranking model.

---

### 2.8 Impressions Distribution Shape

Impressions are **extremely right-skewed**:

| Percentile | Impressions |
|---|---|
| 25th | 81 |
| 50th | 731 |
| 75th | 3,615 |
| 90th | 12,136 |
| 99th | 73,506 |
| Max | 517,715 |

The pipeline handles this correctly via `log_impressions_90d`, `log_clicks_90d`, `log_sessions_90d`.

---

### 2.9 Data Quality Issues to Handle in Feature Engineering

| Issue | Column(s) | Fix |
|---|---|---|
| `avg_position = 0` means no data | `avg_position` | Add `has_position_data` flag; impute 0s with e.g. 100 or median |
| Rate columns ×100 (not ×1) | `ctr`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`, `trend_pct` | **Read as-is — already ×100**. Do NOT multiply again |
| Log-transform needed | `impressions_90d`, `clicks_90d`, `sessions_90d`, `ai_sessions_90d` | `log1p()` — already done in reference pipeline |
| Systematic missingness by `content_type` | `search_volume`, `competition`, `cpc`, `main_intent` | Add `has_keyword_data` indicator; impute with median |
| Missingness by content type for word count | `word_count`, `char_count` | Add `has_word_count` indicator; impute with median |
| Label leakage columns | `trend_direction`, `trend_pct`, `impressions_last_30d`, `clicks_last_30d`, `sessions_last_30d`, `impressions_prev_30d`, `clicks_prev_30d`, `sessions_prev_30d` | **Never features**. Drop before training |
| IDs | `content_id`, `client_id` | Grouping/splitting only; never features |

---

### 2.10 Features Selected by Reference Pipeline (Safe Baseline)

From `scripts/ml_utils.py`:

**Numeric (18):** `search_volume`, `competition`, `cpc`, `word_count`, `char_count`,
`log_impressions_90d`, `log_clicks_90d`, `log_sessions_90d`, `log_ai_sessions_90d`,
`days_with_impressions`, `days_with_sessions`, `content_age_days`, `days_since_last_update`,
`ctr`, `avg_position`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`

**Categorical (8):** `competition_level`, `content_type`, `main_intent`, `age_tier`,
`freshness_tier`, `word_count_tier`, `impression_tier`, `position_tier`

**Top features (by random forest importance):**

| Feature | Importance |
|---|---|
| `days_with_impressions` | 0.160 |
| `log_impressions_90d` | 0.128 |
| `avg_position` | 0.109 |
| `content_age_days` | 0.095 |
| `word_count` | 0.041 |
| `char_count` | 0.040 |

---

### 2.11 Warehouse Data (Week 3+)

The Hugging Face warehouse (`FlyRank/internship-warehouse`) provides the full longitudinal panel
for deeper work:

| Table | Rows | Use |
|---|---|---|
| `dim_clients` | 104 | Client metadata, `gsc_data_start` per client |
| `dim_content` | 519,606 | All content items ever seen |
| `fact_content_daily_performance` | 78.8M | Daily grain — use `_sample` for iteration |
| `fact_content_daily_performance_sample` | ~11.7M | **Use this for all iteration** |
| `fact_content_query_90d` | 2.4M | Per-query context over 90-day window |

> [!CAUTION]
> Key warehouse gotchas:
> - History depth varies per client — always filter on `dim_clients.gsc_data_start` before defining time windows.
> - GA4 columns are zero-filled with `ga4_data_available = FALSE` before the client's `ga4_data_start` — filter on the flag.
> - The query table repeats per-content columns — use `ANY_VALUE()` not `SUM()`.
> - Iterate on `_sample`; run the 79M-row full scan **once** only and cache to `work/outputs/`.

---

## Summary

| CRISP-DM Step | Status | Key Output |
|---|---|---|
| 1 — Business Understanding | ✅ Complete | Decision: rank pages for refresh. Task: ranking via binary classification. Metric: Precision@50. |
| 2 — Data Understanding | ✅ Complete | 30k rows, 44 cols, 54% declining. Key gotchas: rate columns ×100, position=0 means no data, leakage columns identified, missingness is systematic by content type. |
| 3 — Data Preparation | ⬜ Next | Clean, flag missingness, log-transform, encode categoricals, client-holdout split |
| 4 — Modeling | ⬜ Future | Logistic regression → decision tree → random forest; beat Precision@50 = 0.240 |
| 5 — Evaluation | ⬜ Future | Precision@50, ROC-AUC, average precision vs baseline |
| 6 — Deployment | ⬜ Future | Ranked queue output; capstone paper |
