"""Data processing and utility functions for the TWCS dataset."""

from pathlib import Path
from typing import Union
import pandas as pd


def load_twcs(path: Union[str, Path]) -> pd.DataFrame:
    """Load the Twitter Customer Support (TWCS) dataset from a CSV file.

    Expected columns:
        tweet_id, author_id, inbound, created_at, text,
        response_tweet_id, in_response_to_tweet_id

    Args:
        path: Path to the twcs.csv file.

    Returns:
        pd.DataFrame: Loaded TWCS dataframe with standardized column types.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"TWCS data file not found at: {file_path}")

    df = pd.read_csv(
        file_path,
        dtype={
            "author_id": str,
            "text": str,
            "response_tweet_id": str,
        },
    )

    expected_cols = [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ]
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required TWCS columns: {missing}")

    # Standardize inbound as boolean
    if not pd.api.types.is_bool_dtype(df["inbound"]):
        df["inbound"] = (
            df["inbound"].astype(str).str.strip().str.lower().isin(["true", "1"])
        )

    # Standardize tweet IDs as nullable Int64
    df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce").astype("Int64")
    df["in_response_to_tweet_id"] = pd.to_numeric(
        df["in_response_to_tweet_id"], errors="coerce"
    ).astype("Int64")

    return df


def filter_brand(df: pd.DataFrame, brand_handle: str) -> pd.DataFrame:
    """Filter the TWCS dataset for all interactions involving a specific brand.

    Returns all rows involving that brand: both their direct replies
    and the customer tweets they responded to.

    Args:
        df: TWCS dataframe containing customer support tweets.
        brand_handle: Brand Twitter handle (e.g., 'AppleSupport' or '@AppleSupport').

    Returns:
        pd.DataFrame: Filtered dataframe containing the brand's replies and the
            customer tweets they responded to.
    """
    clean_handle = brand_handle.lstrip("@").strip().lower()

    # 1. Brand replies
    brand_mask = df["author_id"].astype(str).str.lower() == clean_handle
    brand_tweets = df[brand_mask]

    # 2. Customer tweets the brand responded to
    responded_to_ids = set(
        pd.to_numeric(brand_tweets["in_response_to_tweet_id"], errors="coerce")
        .dropna()
        .astype("Int64")
    )

    numeric_tweet_ids = pd.to_numeric(df["tweet_id"], errors="coerce").astype("Int64")
    customer_mask = numeric_tweet_ids.isin(responded_to_ids)

    return df[brand_mask | customer_mask].copy().reset_index(drop=True)


def get_resolved_pairs(df: pd.DataFrame, brand_handle: str) -> pd.DataFrame:
    """Extract (customer_tweet, brand_reply) pairs for a specific brand.

    Joins customer inquiry tweets with the brand's corresponding replies,
    forming paired dialogue data used for retrieval grounding and RAG knowledge bases.

    Args:
        df: TWCS dataframe containing customer support tweets.
        brand_handle: Brand Twitter handle (e.g., 'AppleSupport' or '@AppleSupport').

    Returns:
        pd.DataFrame: Dataframe with 'customer_tweet' and 'brand_reply' columns.
    """
    clean_handle = brand_handle.lstrip("@").strip().lower()

    inbound_mask = df["inbound"].astype(str).str.lower().isin(["true", "1"])
    author_mask = df["author_id"].astype(str).str.lower() == clean_handle

    # Outbound brand replies for the specific brand
    brand_replies = df[author_mask & (~inbound_mask)].copy()
    customer_tweets = df[inbound_mask].copy()

    # Convert IDs for reliable numeric join
    brand_replies["in_response_to_tweet_id"] = pd.to_numeric(
        brand_replies["in_response_to_tweet_id"], errors="coerce"
    ).astype("Int64")
    customer_tweets["tweet_id"] = pd.to_numeric(
        customer_tweets["tweet_id"], errors="coerce"
    ).astype("Int64")

    brand_replies = brand_replies.dropna(subset=["in_response_to_tweet_id"])
    customer_tweets = customer_tweets.dropna(subset=["tweet_id"])

    merged = pd.merge(
        customer_tweets[["tweet_id", "text"]],
        brand_replies[["in_response_to_tweet_id", "text"]],
        left_on="tweet_id",
        right_on="in_response_to_tweet_id",
        suffixes=("_customer", "_brand"),
    )

    pairs_df = pd.DataFrame(
        {
            "customer_tweet": merged["text_customer"],
            "brand_reply": merged["text_brand"],
        }
    ).reset_index(drop=True)

    return pairs_df


if __name__ == "__main__":
    # Locate sample TWCS dataset
    sample_path = Path("archive/sample.csv")
    if not sample_path.exists():
        sample_path = Path(__file__).resolve().parent.parent / "archive" / "sample.csv"

    brand = "AppleSupport"
    print(f"Loading TWCS dataset from: {sample_path}")
    raw_df = load_twcs(sample_path)
    print(f"Total dataset row count: {len(raw_df)}")

    brand_df = filter_brand(raw_df, brand)
    print(f"Filtered row count for '{brand}': {len(brand_df)}")

    pairs_df = get_resolved_pairs(raw_df, brand)
    print(f"Resolved pairs row count for '{brand}': {len(pairs_df)}")

    if not pairs_df.empty:
        print("\nFirst resolved pair preview:")
        print(f"  [Customer] : {pairs_df['customer_tweet'].iloc[0]}")
        print(f"  [Reply]    : {pairs_df['brand_reply'].iloc[0]}")
