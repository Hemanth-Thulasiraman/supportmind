# src/data/preprocess.py

import pandas as pd
from loguru import logger
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = Path("data/raw/bitext_raw.parquet")
PROCESSED_DATA_DIR = Path("data/processed")


def get_intent_list(df: pd.DataFrame) -> list[str]:
    """
    Returns sorted list of unique intents from the dataset.
    This list is embedded in every prompt so the model
    knows its valid output space.
    """
    intents = sorted(df["intent"].dropna().unique().tolist())
    logger.info(f"found {len(intents)} unique intents.")
    return intents


def format_prompt(
    instruction: str,
    intent_list: list[str],
    intent: str = None,
    response: str = None
) -> str:
    """
    Formats a prompt for training or inference.
    At training time: pass intent and response → returns full prompt
    At inference time: omit intent and response → returns partial prompt
    for model to complete.

    THIS FUNCTION IS THE SINGLE SOURCE OF TRUTH FOR PROMPT FORMAT.
    Import it in both training and serving. Never duplicate it.
    """
    intent_options = ", ".join(intent_list)

    base = (
        f"<s>[INST] You are a customer support assistant.\n\n"
        f"Given the following ticket, classify the intent and "
        f"generate a helpful response.\n\n"
        f"Ticket: {instruction}\n\n"
        f"Intent categories: {intent_options}\n"
        f"[/INST]\n"
        f"Intent: "
    )

    if intent is not None and response is not None:
        return base + f"{intent}\nResponse: {response}</s>"
    else:
        return base

def create_splits(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into train (80%), validation (10%), test (10%).
    Uses stratified splitting on intent column to ensure
    every intent appears in every split proportionally.
    """
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["intent"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["intent"], random_state=42
    )
    return train_df, val_df, test_df  # no logging here


def run_preprocessing() -> None:
    """
    Main entry point for preprocessing pipeline.
    Loads raw data, formats prompts, creates splits,
    saves each split to data/processed/.
    """
    logger.info("=" * 50)
    logger.info("Starting SupportMind preprocessing")
    logger.info("=" * 50)

    df = pd.read_parquet(RAW_DATA_PATH)
    logger.info(f"Loaded {len(df):,} rows from {RAW_DATA_PATH}")

    intent_list = get_intent_list(df)

    logger.info("Formatting prompts...")
    df["prompt"] = df.apply(
        lambda row: format_prompt(
            instruction=row["instruction"],
            intent_list=intent_list,
            intent=row["intent"],
            response=row["response"]
        ),
        axis=1
    )

    train_df, val_df, test_df = create_splits(df)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(PROCESSED_DATA_DIR / "train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DATA_DIR / "val.parquet", index=False)
    test_df.to_parquet(PROCESSED_DATA_DIR / "test.parquet", index=False)

    logger.info(f"train: {len(train_df):,} rows")
    logger.info(f"val:   {len(val_df):,} rows")
    logger.info(f"test:  {len(test_df):,} rows")
    logger.info(f"Splits saved to {PROCESSED_DATA_DIR}")
    logger.info("Preprocessing complete")

if __name__ == "__main__":
    run_preprocessing()