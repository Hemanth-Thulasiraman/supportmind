# src/evaluation/metrics.py

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_intent_f1(
    y_true: list[str],
    y_pred: list[str],
) -> dict:
    """
    Computes macro F1, weighted F1, and per-class F1
    for intent classification.
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    logger.info(f"Macro F1:    {macro_f1:.4f}")
    logger.info(f"Weighted F1: {weighted_f1:.4f}")

    per_class = {
        k: v["f1-score"]
        for k, v in report.items()
        if k not in ["accuracy", "macro avg", "weighted avg"]
    }
    sorted_classes = sorted(per_class.items(), key=lambda x: x[1])

    logger.info("5 worst performing intents:")
    for intent, f1 in sorted_classes[:5]:
        logger.info(f"  {intent:<40} F1={f1:.4f}")

    logger.info("5 best performing intents:")
    for intent, f1 in sorted_classes[-5:]:
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
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    logger.info("Confusion matrix computed")
    return df_cm


def compute_rouge(
    predictions: list[str],
    references: list[str],
) -> dict:
    """
    Computes ROUGE-1 and ROUGE-L for generated responses.
    More reliable than BERTScore across library versions.
    """
    try:
        import evaluate as hf_evaluate
        rouge = hf_evaluate.load("rouge")
        results = rouge.compute(
            predictions=predictions,
            references=references,
        )
        logger.info(f"ROUGE-1: {results['rouge1']:.4f}")
        logger.info(f"ROUGE-L: {results['rougeL']:.4f}")
        return results
    except Exception as e:
        logger.warning(f"ROUGE computation failed: {e}")
        return {}


def extract_intent_from_output(
    model_output: str,
    valid_intents: list[str] = None,
) -> str:
    """
    Robust parser for model output.
    Handles three formats the model might generate:
    1. 'Intent: cancel_order\\nResponse: ...'
    2. 'cancel_order\\nResponse: ...'  (no prefix — our model does this)
    3. Intent label appears in first 50 chars
    """
    if not model_output or not model_output.strip():
        return "unknown"

    text = model_output.strip()

    # Strategy 1 — Look for "Intent: X" prefix
    for line in text.split("\n"):
        line = line.strip()
        if line.lower().startswith("intent:"):
            intent = line.split(":", 1)[1].strip()
            intent = intent.split("(")[0].strip()
            intent = intent.lower().replace(" ", "_").replace("-", "_")
            if intent:
                return intent

    # Strategy 2 — First line IS the intent (no prefix)
    # Our model does this: 'cancel_order\nResponse: ...'
    first_line = text.split("\n")[0].strip()
    normalized = first_line.lower().replace(" ", "_").replace("-", "_")
    if valid_intents and normalized in valid_intents:
        return normalized

    # Strategy 3 — Valid intent appears in first 50 chars
    if valid_intents:
        text_start = text[:50].lower()
        for intent in valid_intents:
            if intent in text_start:
                return intent

    return "unknown"


def extract_response_from_output(model_output: str) -> str:
    """
    Extracts generated response from model output.
    Handles both 'Response:' prefix and plain text formats.
    """
    if not model_output:
        return ""
    try:
        if "Response:" in model_output:
            response = model_output.split("Response:", 1)[1].strip()
            response = response.replace("</s>", "").strip()
            return response
        # Fallback: return everything after first line
        lines = model_output.strip().split("\n")
        if len(lines) > 1:
            return "\n".join(lines[1:]).strip()
    except Exception:
        pass
    return ""