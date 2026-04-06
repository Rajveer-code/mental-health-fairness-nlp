"""
preprocess.py  [FIXED — v2]
-------------
Loads, cleans, and prepares all datasets for training and fairness audit.

CRITICAL FIX FROM ORIGINAL:
  GoEmotions is a MULTI-LABEL dataset — each example can have 2–3 emotion
  labels simultaneously.  The original code used row["labels"][0], which
  takes the LOWEST label ID (not the dominant label), introducing selection
  bias and silently discarding co-occurring emotion information.

  FIX — _resolve_goemotions_label():
    For single-label examples: use that label directly (after mapping).
    For multi-label examples: map ALL non-ambiguous labels.
      - If all map to the SAME unified class → use that class (unambiguous).
      - If they map to DIFFERENT classes    → DROP the example (ambiguous).
      - If ALL map to None (ambiguous emotions) → DROP.

  This is methodologically correct: we only include examples where the
  annotation is unambiguous under our four-class schema.  The resulting
  GoEmotions corpus will be slightly smaller (ambiguous multi-label
  examples dropped) but more reliable.

All other logic (Kaggle mapping, Twitter mapping, splits, cleaning)
is unchanged from the original.
"""

import os
import re
import yaml
import pandas as pd
import numpy as np
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────

with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

SEED = cfg["training"]["seed"]
np.random.seed(SEED)

# ── Label maps ────────────────────────────────────────────────────────────────

KAGGLE_TO_UNIFIED = {
    "Normal":               "normal",
    "Depression":           "depression",
    "Suicidal":             "depression",
    "Anxiety":              "anxiety",
    "Bipolar":              "depression",
    "Stress":               "stress",
    "Personality disorder": "stress",
}

GOEMO_TO_UNIFIED = {
    "sadness":        "depression",
    "grief":          "depression",
    "remorse":        "depression",
    "disappointment": "depression",
    "fear":           "anxiety",
    "nervousness":    "anxiety",
    "embarrassment":  "anxiety",
    "anger":          "stress",
    "annoyance":      "stress",
    "disgust":        "stress",
    "joy":            "normal",
    "love":           "normal",
    "optimism":       "normal",
    "gratitude":      "normal",
    "relief":         "normal",
    "admiration":     "normal",
    "amusement":      "normal",
    "approval":       "normal",
    "caring":         "normal",
    "confusion":      None,    # ambiguous — drop
    "curiosity":      None,
    "desire":         None,
    "excitement":     "normal",
    "pride":          "normal",
    "realization":    None,
    "surprise":       None,
    "neutral":        "normal",
}

DAIREMO_TO_UNIFIED = {
    0: "depression",   # sadness
    1: "normal",       # joy
    2: "normal",       # love
    3: "stress",       # anger
    4: "anxiety",      # fear
    5: None,           # surprise — drop
}

LABEL_TO_INT = {
    "normal":     0,
    "depression": 1,
    "anxiety":    2,
    "stress":     3,
}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


# ── Text cleaning (unchanged) ─────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.encode("ascii", "ignore").decode("ascii")
    if len(text.split()) < 4:
        return ""
    return text


# ── Demographic proxy extraction (unchanged) ──────────────────────────────────

AGE_YOUNG_KEYWORDS = [
    "college", "university", "uni", "school", "studying", "student",
    "exam", "semester", "dorm", "freshman", "sophomore", "teen",
    "tiktok", "instagram", "snap", "discord", "gaming", "parents wont",
    "my parents", "mom wont", "dad wont", "high school"
]
AGE_OLDER_KEYWORDS = [
    "career", "job loss", "mortgage", "retirement", "pension",
    "grandchild", "decades", "my kids", "my children", "divorce",
    "been married", "years of marriage", "work stress", "promotion",
    "layoff", "fired from", "decades ago", "getting old", "aging"
]
GENDER_MALE_KEYWORDS = [
    "my wife", "my girlfriend", "as a man", "as a guy", "i am a man",
    "being a man", "my daughter", "my son", "brotherhood", "masculin"
]
GENDER_FEMALE_KEYWORDS = [
    "my husband", "my boyfriend", "as a woman", "as a girl",
    "i am a woman", "being a woman", "my period", "pregnancy",
    "sisterhood", "feminin", "postpartum"
]


def infer_age_group(text: str) -> str:
    text_lower  = text.lower()
    young_score = sum(1 for kw in AGE_YOUNG_KEYWORDS if kw in text_lower)
    older_score = sum(1 for kw in AGE_OLDER_KEYWORDS if kw in text_lower)
    if young_score > older_score and young_score > 0:
        return "young"
    elif older_score > young_score and older_score > 0:
        return "older"
    return "unknown"


def infer_gender(text: str) -> str:
    text_lower   = text.lower()
    male_score   = sum(1 for kw in GENDER_MALE_KEYWORDS   if kw in text_lower)
    female_score = sum(1 for kw in GENDER_FEMALE_KEYWORDS if kw in text_lower)
    if male_score > female_score and male_score > 0:
        return "male"
    elif female_score > male_score and female_score > 0:
        return "female"
    return "unknown"


# ── FIXED: GoEmotions multi-label resolver ────────────────────────────────────

def _resolve_goemotions_label(label_ids: list,
                              label_names: list,
                              mapping: dict) -> str | None:
    """
    Resolve a (potentially multi-label) GoEmotions annotation to a single
    unified four-class label.

    Strategy:
      1. Map every non-ambiguous label to its unified class.
      2. If all mapped classes agree → return that class.
      3. If mapped classes conflict, or if ALL labels are ambiguous (None)
         → return None (example will be dropped).

    This replaces the original row["labels"][0] approach, which used the
    lowest label ID rather than the semantically dominant one, and silently
    discarded multi-label information.

    Parameters
    ----------
    label_ids : list[int]
        List of GoEmotions label indices for this example.
    label_names : list[str]
        Full list of GoEmotions label names (from the dataset features).
    mapping : dict
        GOEMO_TO_UNIFIED mapping.

    Returns
    -------
    str or None
        Unified class string, or None if ambiguous.
    """
    if not label_ids:
        return None

    # Map each label to its unified class (None = ambiguous/excluded)
    mapped_classes = [
        mapping.get(label_names[lid])
        for lid in label_ids
    ]

    # Keep only non-None mappings
    resolved = [c for c in mapped_classes if c is not None]

    if not resolved:
        # All labels are in the "drop" category (confusion, curiosity, etc.)
        return None

    unique_classes = set(resolved)

    if len(unique_classes) == 1:
        # All non-ambiguous labels agree on the same class → unambiguous
        return unique_classes.pop()

    # Labels map to different classes → ambiguous, drop
    return None


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_kaggle(path: str = "data/raw/kaggle_mental_health/Combined Data.csv") -> pd.DataFrame:
    """Load and clean primary Kaggle mental health dataset. Unchanged."""
    print("\n[1/3] Loading Kaggle Mental Health dataset...")
    df = pd.read_csv(path, usecols=["statement", "status"])
    df.columns = ["text", "label_raw"]

    df["label_str"] = df["label_raw"].map(KAGGLE_TO_UNIFIED)
    df = df.dropna(subset=["label_str"])

    df["text"]  = df["text"].apply(clean_text)
    df = df[df["text"] != ""]

    df["label"]    = df["label_str"].map(LABEL_TO_INT)
    df["platform"] = "multi"

    df["age_group"] = df["text"].apply(infer_age_group)
    df["gender"]    = df["text"].apply(infer_gender)

    print(f"  Loaded: {len(df):,} samples")
    print(f"  Label distribution:\n{df['label_str'].value_counts().to_string()}")
    return df[["text", "label", "label_str", "platform", "age_group", "gender"]]


def load_goemotions(path: str = "data/raw/reddit_goemotions") -> pd.DataFrame:
    """
    Load GoEmotions Reddit dataset and remap to unified 4-class schema.

    FIXED: Uses _resolve_goemotions_label() instead of labels[0].
    Multi-label examples with conflicting classes are dropped.
    Single-label and unambiguous multi-label examples are retained.
    """
    print("\n[2/3] Loading GoEmotions Reddit dataset (FIXED multi-label handling)...")
    ds = load_from_disk(path)

    label_names = ds["train"].features["labels"].feature.names

    rows            = []
    n_total         = 0
    n_single        = 0
    n_multi_kept    = 0
    n_multi_dropped = 0
    n_ambiguous     = 0

    for split in ["test"]: 
        for row in ds[split]:
            n_total += 1
            label_ids = row["labels"]

            if not label_ids:
                n_ambiguous += 1
                continue

            # FIXED: resolve multi-label correctly
            unified = _resolve_goemotions_label(label_ids, label_names,
                                                GOEMO_TO_UNIFIED)

            if unified is None:
                if len(label_ids) == 1:
                    n_ambiguous += 1
                else:
                    n_multi_dropped += 1
                continue

            if len(label_ids) == 1:
                n_single += 1
            else:
                n_multi_kept += 1

            rows.append({
                "text":      row["text"],
                "label_str": unified,
                "label":     LABEL_TO_INT[unified],
                "platform":  "reddit",
            })

    df = pd.DataFrame(rows)
    df["text"]      = df["text"].apply(clean_text)
    df = df[df["text"] != ""]
    df["age_group"] = df["text"].apply(infer_age_group)
    df["gender"]    = df["text"].apply(infer_gender)

    print(f"  Total examples processed:          {n_total:,}")
    print(f"  Single-label → retained:           {n_single:,}")
    print(f"  Multi-label (unanimous) → retained:{n_multi_kept:,}")
    print(f"  Multi-label (conflicting) → dropped:{n_multi_dropped:,}")
    print(f"  Ambiguous labels → dropped:        {n_ambiguous:,}")
    print(f"  After text cleaning:               {len(df):,} samples")
    print(f"  Label distribution:\n{df['label_str'].value_counts().to_string()}")

    return df[["text", "label", "label_str", "platform", "age_group", "gender"]]


def load_twitter_emotion(path: str = "data/raw/twitter_emotion") -> pd.DataFrame:
    """Load dair-ai/emotion Twitter dataset. Unchanged."""
    print("\n[3/3] Loading Twitter Emotion dataset...")
    ds = load_from_disk(path)

    rows = []
    for split in ["test"]:
        for row in ds[split]:
            unified = DAIREMO_TO_UNIFIED.get(row["label"])
            if unified is None:
                continue
            rows.append({
                "text":      row["text"],
                "label_str": unified,
                "label":     LABEL_TO_INT[unified],
                "platform":  "twitter",
            })

    df = pd.DataFrame(rows)
    df["text"]      = df["text"].apply(clean_text)
    df = df[df["text"] != ""]
    df["age_group"] = df["text"].apply(infer_age_group)
    df["gender"]    = df["text"].apply(infer_gender)

    print(f"  Loaded: {len(df):,} samples")
    print(f"  Label distribution:\n{df['label_str'].value_counts().to_string()}")
    return df[["text", "label", "label_str", "platform", "age_group", "gender"]]


# ── Split and save (unchanged) ────────────────────────────────────────────────

def split_and_save(df: pd.DataFrame, name: str) -> None:
    out_dir = os.path.join(cfg["paths"]["splits"], name)
    os.makedirs(out_dir, exist_ok=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED
    )

    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val_df.to_csv(  os.path.join(out_dir, "val.csv"),   index=False)
    test_df.to_csv( os.path.join(out_dir, "test.csv"),  index=False)

    print(f"\n  Saved splits for [{name}]:")
    print(f"    Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"    Location: {out_dir}/")


def print_summary(dfs: dict) -> None:
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE — DATASET SUMMARY")
    print("="*60)
    total = 0
    for name, df in dfs.items():
        n = len(df)
        total += n
        print(f"\n  {name.upper()} ({n:,} samples)")
        print(f"  Platform: {df['platform'].unique()}")
        label_pct = (df["label_str"].value_counts(normalize=True) * 100).round(1)
        for lbl, pct in label_pct.items():
            print(f"    {lbl:<20} {pct}%")
    print(f"\n  TOTAL SAMPLES: {total:,}")
    print("="*60)


def create_cross_platform_splits() -> None:
    out_dir = os.path.join(cfg["paths"]["splits"], "cross_platform")
    os.makedirs(out_dir, exist_ok=True)

    kaggle_train = pd.read_csv("data/splits/kaggle/train.csv")
    kaggle_val   = pd.read_csv("data/splits/kaggle/val.csv")
    kaggle_test  = pd.read_csv("data/splits/kaggle/test.csv")
    reddit_test  = pd.read_csv("data/splits/reddit/test.csv")
    twitter_test = pd.read_csv("data/splits/twitter/test.csv")

    kaggle_train.to_csv(os.path.join(out_dir, "train.csv"),        index=False)
    kaggle_val.to_csv(  os.path.join(out_dir, "val.csv"),          index=False)
    kaggle_test.to_csv( os.path.join(out_dir, "test_kaggle.csv"),  index=False)
    reddit_test.to_csv( os.path.join(out_dir, "test_reddit.csv"),  index=False)
    twitter_test.to_csv(os.path.join(out_dir, "test_twitter.csv"), index=False)

    print("\n" + "="*60)
    print("CROSS-PLATFORM EVALUATION SETUP")
    print("="*60)
    print(f"  Train (Kaggle):         {len(kaggle_train):,}")
    print(f"  Validation (Kaggle):    {len(kaggle_val):,}")
    print(f"  Test - Within platform: {len(kaggle_test):,}")
    print(f"  Test - Reddit:          {len(reddit_test):,}")
    print(f"  Test - Twitter:         {len(twitter_test):,}")
    print(f"\n  Saved to: {out_dir}/")

    for name, df in [("Kaggle test", kaggle_test),
                     ("Reddit test", reddit_test),
                     ("Twitter test", twitter_test)]:
        print(f"\n  {name} label distribution:")
        dist = (df["label_str"].value_counts(normalize=True) * 100).round(1)
        for lbl, pct in dist.items():
            print(f"    {lbl:<20} {pct}%")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting preprocessing pipeline (FIXED — correct multi-label handling)...")
    print(f"Random seed: {SEED}")
    print(f"Label schema: {LABEL_TO_INT}")

    df_kaggle  = load_kaggle()
    df_reddit  = load_goemotions()   # FIXED
    df_twitter = load_twitter_emotion()

    split_and_save(df_kaggle,  "kaggle")
    split_and_save(df_reddit,  "reddit")
    split_and_save(df_twitter, "twitter")

    df_combined = pd.concat([df_kaggle, df_reddit], ignore_index=True)
    split_and_save(df_combined, "combined")

    create_cross_platform_splits()

    print_summary({
        "kaggle":  df_kaggle,
        "reddit":  df_reddit,
        "twitter": df_twitter,
    })

    print("\nAll done.")
    print("NOTE: If GoEmotions sample count changed from the original preprocessing,")
    print("      re-run: python src/train.py --model all")
    print("      then:   python src/evaluate.py")
    print("      to regenerate all downstream outputs with the corrected data.")
    print("\nNext step: python src/train.py --model all")
