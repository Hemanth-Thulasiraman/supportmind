# src/data/ingest.py

from datasets import load_dataset
from loguru import logger
from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path("data/raw")
DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
EXPECTED_COLUMNS = ["instruction", "intent", "response"]
MAX_CLASS_DOMINANCE = 0.20


def download_dataset() -> pd.DataFrame:
    """
    Downloads the Bitext dataset from HuggingFace.
    Saves parquet copy to data/raw/bitext_raw.parquet.
    Returns DataFrame.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    # Check available splits before assuming 'train' exists
    available_splits = list(dataset.keys())
    logger.info(f"Available splits: {available_splits}")

    if "train" not in dataset:
        raise ValueError(
            f"Expected 'train' split not found. "
            f"Available splits: {available_splits}"
        )

    df = dataset["train"].to_pandas()

    # Warn if expected columns are missing
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        logger.warning(f"Missing expected columns: {missing}")

    output_path = RAW_DATA_DIR / "bitext_raw.parquet"
    df.to_parquet(output_path, index=False)

    logger.info(f"Rows: {len(df):,}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Saved to: {output_path}")

    return df


def log_label_distribution(df: pd.DataFrame) -> None:
    """
    Logs intent label distribution.
    Warns if any single intent exceeds MAX_CLASS_DOMINANCE.
    """
    intent_dist = df["intent"].value_counts(normalize=True).sort_index()

    logger.info("Intent label distribution:")
    for intent, proportion in intent_dist.items():
        pct = proportion * 100
        flag = " ⚠️  EXCEEDS 20%" if proportion > MAX_CLASS_DOMINANCE else ""
        logger.info(f"  {intent:<40} {pct:5.1f}%{flag}")

    dominant = intent_dist[intent_dist > MAX_CLASS_DOMINANCE]
    if not dominant.empty:
        logger.warning(
            f"{len(dominant)} intent(s) exceed {MAX_CLASS_DOMINANCE:.0%} "
            f"dominance threshold — check for class imbalance"
        )
    else:
        logger.info("Class dominance check passed — no intent exceeds 20%")


def run_ingestion() -> pd.DataFrame:
    """
    Main entry point for data ingestion pipeline.
    Downloads, saves, and profiles the raw dataset.
    Returns the loaded DataFrame.
    """
    logger.info("=" * 50)
    logger.info("Starting SupportMind data ingestion")
    logger.info("=" * 50)

    df = download_dataset()
    log_label_distribution(df)

    logger.info("Ingestion complete")
    return df


if __name__ == "__main__":
    run_ingestion()