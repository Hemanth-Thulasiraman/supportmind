# src/evaluation/metrics.py

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from typing import Optional


def compute_intent_f1(
    y_true: list[str],
    y_pred: list[str],
) -> dict:
    """
    Computes macro F1, weighted F1, and per-class F1
    for intent classification.
    Returns a dict of metrics.
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # Per-class F1
    report = classification_report(
        y_true, y_pred,
        output_dict=True,
        zero_division=0
    )

    logger.info(f"Macro F1:    {macro_f1:.4f}")
    logger.info(f"Weighted F1: {weighted_f1:.4f}")

    # Log bottom 5 worst performing intents
    per_class = {
        k: v["f1-score"]
        for k, v in report.items()
        if k not in ["accuracy", "macro avg", "weighted avg"]
    }
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1])
    logger.info("5 worst performing intents:")
    for intent, f1 in sorted_classes[:5]:
        logger.info(f"  {intent:<40} F1={f1:.4f}")

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class,
        "classification_report": report,
    }


def compute_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> pd.DataFrame:
    """
    Computes confusion matrix as a labeled DataFrame.
    Useful for identifying which intents get confused with each other.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    logger.info("Confusion matrix computed")
    return df_cm


def compute_bert_score(
    predictions: list[str],
    references: list[str],
    sample_size: int = 200,
) -> dict:
    """
    Computes BERTScore for generated responses vs reference responses.
    Samples a subset for speed — full BERTScore on 2688 examples is slow.
    Returns mean precision, recall, F1.
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        logger.warning("bert_score not installed — skipping BERTScore")
        return {}

    # Sample for speed
    if len(predictions) > sample_size:
        indices = np.random.choice(
            len(predictions), sample_size, replace=False
        )
        predictions = [predictions[i] for i in indices]
        references = [references[i] for i in indices]
        logger.info(f"BERTScore: sampling {sample_size} examples for speed")

    logger.info("Computing BERTScore — this takes 1-2 minutes...")
    P, R, F1 = bert_score_fn(
        predictions,
        references,
        lang="en",
        verbose=False,
    )

    results = {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }

    logger.info(f"BERTScore Precision: {results['bertscore_precision']:.4f}")
    logger.info(f"BERTScore Recall:    {results['bertscore_recall']:.4f}")
    logger.info(f"BERTScore F1:        {results['bertscore_f1']:.4f}")

    return results


def extract_intent_from_output(model_output: str) -> str:
    """
    Parses raw model output to extract the intent label.
    Model output format:
        'Intent: cancel_order\\nResponse: ...'
    Returns the intent string or 'unknown' if parsing fails.
    """
    try:
        lines = model_output.strip().split("\n")
        for line in lines:
            if line.startswith("Intent:"):
                intent = line.replace("Intent:", "").strip()
                return intent
    except Exception:
        pass
    return "unknown"


def extract_response_from_output(model_output: str) -> str:
    """
    Parses raw model output to extract the generated response.
    Returns the response string or empty string if parsing fails.
    """
    try:
        if "Response:" in model_output:
            response = model_output.split("Response:", 1)[1].strip()
            # Remove </s> end token if present
            response = response.replace("</s>", "").strip()
            return response
    except Exception:
        pass
    return ""