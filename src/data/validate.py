# src/data/validate.py

import pandas as pd
from loguru import logger

REQUIRED_COLUMNS = ["instruction", "intent", "response"]
MIN_INTENTS = 20
MAX_INTENTS = 35
MAX_CLASS_DOMINANCE = 0.20


def check_required_columns(df: pd.DataFrame) -> bool:
    """Returns True if all required columns are present."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return False
    logger.info(f"✅ Required columns check passed: {REQUIRED_COLUMNS}")
    return True


def check_no_nulls(df: pd.DataFrame) -> bool:
    """Returns True if no nulls exist in required columns."""
    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    has_nulls = null_counts[null_counts > 0]
    if not has_nulls.empty:
        logger.error(f"Null values found: {has_nulls.to_dict()}")
        return False
    logger.info("✅ Null check passed: no nulls in required columns")
    return True


def check_intent_count(df: pd.DataFrame) -> bool:
    """Returns True if unique intent count is within expected range."""
    n_intents = df["intent"].nunique()
    if not (MIN_INTENTS <= n_intents <= MAX_INTENTS):
        logger.error(
            f"Intent count {n_intents} outside expected range "
            f"[{MIN_INTENTS}, {MAX_INTENTS}]"
        )
        return False
    logger.info(f"✅ Intent count check passed: {n_intents} unique intents")
    return True


def check_class_dominance(df: pd.DataFrame) -> bool:
    """Returns True if no single class exceeds MAX_CLASS_DOMINANCE."""
    dist = df["intent"].value_counts(normalize=True)
    dominant = dist[dist > MAX_CLASS_DOMINANCE]
    if not dominant.empty:
        logger.error(
            f"Dominant intents found (exceed {MAX_CLASS_DOMINANCE:.0%}): "
            f"{dominant.to_dict()}"
        )
        return False
    logger.info(
        f"✅ Class dominance check passed: "
        f"max class proportion = {dist.max():.1%}"
    )
    return True


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Runs all validation checks. Logs pass/fail for each.
    Raises ValueError listing ALL failures if any check fails.
    Returns True if all checks pass.
    """
    logger.info("Running dataset validation...")

    checks = {
        "required_columns": check_required_columns,
        "no_nulls": check_no_nulls,
        "intent_count": check_intent_count,
        "class_dominance": check_class_dominance,
    }

    # Run ALL checks and collect failures
    failures = []
    for check_name, check_fn in checks.items():
        if not check_fn(df):
            failures.append(check_name)

    if failures:
        raise ValueError(
            f"Dataset validation failed. "
            f"Failed checks: {failures}. "
            f"Fix these before proceeding."
        )

    logger.info("✅ All validation checks passed")
    return True


if __name__ == "__main__":
    df = pd.read_parquet("data/raw/bitext_raw.parquet")
    validate_dataset(df)