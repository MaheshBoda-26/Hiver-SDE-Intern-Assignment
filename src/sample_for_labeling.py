"""Sample customer tweets from AmazonHelp resolved pairs for manual intent labeling."""

from pathlib import Path
import pandas as pd

from src.data_utils import load_twcs, get_resolved_pairs

# ── Configuration ────────────────────────────────────────────────────────────
BRAND_HANDLE = "AmazonHelp"
SAMPLE_SIZE = 300
RANDOM_SEED = 42
MIN_CHAR_LENGTH = 15  # Exclude very short tweets (noise, single emoji, etc.)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "twcs_sample.csv",
    PROJECT_ROOT / "archive" / "twcs" / "twcs_sample.csv",
    PROJECT_ROOT / "archive" / "twcs" / "twcs.csv",
    PROJECT_ROOT / "archive" / "sample.csv",
]
OUTPUT_PATH = PROJECT_ROOT / "eval" / "labeling_sample.csv"


def find_data_path() -> Path:
    """Locate the first available TWCS dataset file."""
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find TWCS dataset. Searched:\n"
        + "\n".join(f"  - {p}" for p in DATA_CANDIDATES)
    )


def main() -> None:
    # Step 1: Load dataset and extract resolved pairs
    data_path = find_data_path()
    print(f"Loading TWCS dataset from: {data_path}")
    raw_df = load_twcs(data_path)
    print(f"Total dataset rows: {len(raw_df):,}")

    pairs_df = get_resolved_pairs(raw_df, BRAND_HANDLE)
    print(f"Resolved pairs for '{BRAND_HANDLE}': {len(pairs_df):,}")

    # Step 2: Filter out short tweets (likely noise)
    pairs_df["char_length"] = pairs_df["customer_tweet"].str.len()
    before_filter = len(pairs_df)
    pairs_df = pairs_df[pairs_df["char_length"] >= MIN_CHAR_LENGTH].copy()
    filtered_out = before_filter - len(pairs_df)
    print(
        f"Filtered out {filtered_out:,} tweets under {MIN_CHAR_LENGTH} chars "
        f"→ {len(pairs_df):,} remaining"
    )

    # Step 3: Random sample (reproducible)
    n_sample = min(SAMPLE_SIZE, len(pairs_df))
    sample_df = pairs_df.sample(n=n_sample, random_state=RANDOM_SEED).reset_index(
        drop=True
    )

    # Step 4: Prepare output columns
    output_df = sample_df[["tweet_id", "customer_tweet", "brand_reply"]].rename(
        columns={
            "customer_tweet": "customer_text",
            "brand_reply": "brand_reply_text",
        }
    )

    # Step 5: Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(output_df):,} samples → {OUTPUT_PATH}")

    # Step 6: Print examples
    n_preview = min(5, len(output_df))
    print(f"\n{'='*80}")
    print(f"Preview ({n_preview} of {len(output_df):,} samples):")
    print(f"{'='*80}")
    for idx in range(n_preview):
        row = output_df.iloc[idx]
        print(f"\n--- Sample #{idx + 1} (tweet_id: {row['tweet_id']}) ---")
        print(f"  [Customer] : {row['customer_text']}")
        print(f"  [Reply]    : {row['brand_reply_text']}")


if __name__ == "__main__":
    main()
