# Data

This directory contains only **split index files** — not raw text corpora.
Raw data must be downloaded separately (see instructions below) because:
1. The Kaggle dataset requires a Kaggle account and API key
2. The HuggingFace datasets are large (combined ~500 MB)
3. GoEmotions and dair-ai/emotion have their own license terms (CC BY 4.0 / MIT)

---

## Datasets Used

### 1. Training Platform — Kaggle Combined Mental Health Corpus

| Property | Value |
|----------|-------|
| Source | Sarkar (2022), HuggingFace Datasets (Kaggle) |
| Original classes | 7 (Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality Disorder) |
| After remapping | 4 (normal, depression, anxiety, stress) |
| Training split | n = 35,556 (70%) |
| Validation split | n = 7,620 (15%) |
| Within-platform test | n = 7,620 (15%, stratified) |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

**Download:**
```bash
# Install Kaggle CLI
pip install kaggle

# Configure API key: place kaggle.json in ~/.kaggle/kaggle.json
# Download the dataset
kaggle datasets download suchintikasarkar/sentiment-analysis-for-mental-health
unzip sentiment-analysis-for-mental-health.zip -d data/raw/kaggle_mental_health/
```

**Label remapping:**
```
Normal               → normal
Depression           → depression
Suicidal             → depression  (suicidal ideation: severe depression cluster)
Anxiety              → anxiety
Bipolar              → depression  (mood disorder cluster)
Stress               → stress
Personality disorder → stress      (distress cluster)
```

---

### 2. Cross-Platform Test Set 1 — GoEmotions (Reddit)

| Property | Value |
|----------|-------|
| Source | Demszky et al. (2020); HuggingFace `go_emotions` |
| Original classes | 27 fine-grained emotions + neutral |
| After remapping | 4 (normal, depression, anxiety, stress); ambiguous classes dropped |
| Cross-platform test | n = 6,257 (held-out test partition after mapping + drop) |
| License | [Apache 2.0](https://github.com/google-research/google-research/blob/master/LICENSE) |

**Download (automatic):**
```bash
python scripts/01_download_data.py
# OR manually:
python -c "from datasets import load_dataset; load_dataset('go_emotions', split='test').save_to_disk('data/raw/reddit_goemotions/test')"
```

**Label remapping:**
```
sadness, grief, remorse, disappointment → depression
fear, nervousness, embarrassment        → anxiety
anger, annoyance, disgust               → stress
joy, love, optimism, gratitude, relief,
  admiration, amusement, approval,
  caring, excitement, pride, neutral    → normal
confusion, curiosity, desire,
  realization, surprise                 → dropped (ambiguous)
```

> ⚠️ **Non-independence warning (GoEmotions-RoBERTa):** The `mentalroberta`
> model (SamLowe/roberta-base-go_emotions) was fine-tuned on the full
> GoEmotions corpus — the same source as this test set. Its Reddit results
> are therefore **non-independent** and represent an in-distribution
> performance ceiling, not a cross-platform benchmark. Flagged with † in
> all tables.

---

### 3. Cross-Platform Test Set 2 — dair-ai/emotion (Twitter)

| Property | Value |
|----------|-------|
| Source | Saravia et al. (2018); HuggingFace `dair-ai/emotion` |
| Original classes | 6 (sadness, joy, love, anger, fear, surprise) |
| After remapping | 4 (normal, depression, anxiety, stress); surprise dropped |
| Cross-platform test | n = 2,883 (validation + test partitions after mapping) |
| License | [MIT](https://github.com/dair-ai/emotion_dataset) |

**Download (automatic):**
```bash
python scripts/01_download_data.py
```

**Label remapping:**
```
sadness  → depression
joy      → normal
love     → normal
anger    → stress
fear     → anxiety
surprise → dropped (no clear clinical proxy)
```

---

## Directory Structure

```
data/
├── README.md                 ← this file
├── raw/                      ← gitignored; download with 01_download_data.py
│   ├── kaggle_mental_health/ ← Combined Data.csv
│   ├── reddit_goemotions/    ← HuggingFace arrow files
│   └── twitter_emotion/      ← HuggingFace arrow files
└── splits/
    └── cross_platform/       ← committed; produced by preprocess.py
        ├── train.csv          ← n=35,556 Kaggle training samples
        ├── val.csv            ← n=7,620 Kaggle validation samples
        ├── test_kaggle.csv    ← n=7,620 within-platform test
        ├── test_reddit.csv    ← n=6,257 cross-platform test (GoEmotions)
        └── test_twitter.csv   ← n=2,883 cross-platform test (dair-ai/emotion)
```

All split CSVs have columns: `text, label` where label ∈ {0=normal, 1=depression, 2=anxiety, 3=stress}.

---

## Regenerating Splits

If you need to regenerate the splits from raw data:

```bash
python src/preprocess.py
```

This will download GoEmotions and dair-ai/emotion from HuggingFace automatically.
For the Kaggle dataset, you must first download `Combined Data.csv` manually
and place it at `data/raw/kaggle_mental_health/Combined Data.csv`.

The committed splits in `data/splits/cross_platform/` were generated with
`seed=42` and match the paper exactly.

---

## Label Distribution

| Class | Label ID | Kaggle train % | Kaggle test % | Reddit test % | Twitter test % |
|-------|---------|---------------|--------------|--------------|---------------|
| normal | 0 | 28.7% | 28.7% | 45.1% | 43.6% |
| depression | 1 | 56.6% | 56.6% | 16.1% | 30.1% |
| anxiety | 2 | 7.5% | 7.5% | 19.0% | 12.3% |
| stress | 3 | 7.2% | 7.2% | 19.8% | 14.0% |

The substantial label distribution shift between Kaggle and the cross-platform
test sets is an independent confound in addition to covariate shift (Section 5.5).
