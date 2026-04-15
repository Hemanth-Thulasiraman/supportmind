# src/monitoring/drift.py

import json
import pandas as pd
from datetime import datetime
from loguru import logger
from pathlib import Path
from src.monitoring.logger import load_prediction_logs

REPORTS_DIR = Path("data/monitoring/reports")


def compute_intent_distribution(
    records: list[dict],
) -> pd.Series:
    """
    Computes intent distribution from prediction logs.
    Returns normalized value counts.
    """
    df = pd.DataFrame(records)
    return df["predicted_intent"].value_counts(normalize=True)


def check_confidence_degradation(
    records: list[dict],
    low_confidence_threshold: float = 0.10,
) -> dict:
    """
    Flags if low confidence predictions exceed threshold.
    More than 10% low confidence = model is struggling.
    """
    df = pd.DataFrame(records)
    low_conf_rate = float((df["confidence"] == "low").mean())

    result = {
        "low_confidence_rate": round(low_conf_rate, 4),
        "threshold": low_confidence_threshold,
        "alert": bool(low_conf_rate > low_confidence_threshold),
    }

    if result["alert"]:
        logger.warning(
            f"HIGH LOW-CONFIDENCE RATE: "
            f"{low_conf_rate:.1%} of predictions have low confidence "
            f"(threshold: {low_confidence_threshold:.0%}). "
            f"Model may be struggling with input distribution."
        )
    else:
        logger.info(
            f"Confidence check passed: "
            f"{low_conf_rate:.1%} low confidence rate"
        )

    return result


def check_intent_distribution_shift(
    current_records: list[dict],
    reference_path: str = "data/processed/train.parquet",
) -> dict:
    """
    Compares current prediction distribution against
    training data distribution.
    Flags if any intent shifts by more than 5 percentage points.
    """
    try:
        train_df = pd.read_parquet(reference_path)
        reference_dist = train_df["intent"].value_counts(
            normalize=True
        )
    except Exception as e:
        logger.warning(f"Could not load reference distribution: {e}")
        return {"error": str(e)}

    current_dist = compute_intent_distribution(current_records)

    shifts = {}
    alerts = []

    for intent in reference_dist.index:
        ref = float(reference_dist.get(intent, 0))
        cur = float(current_dist.get(intent, 0))
        shift = abs(cur - ref)
        alert = bool(shift > 0.05)
        shifts[intent] = {
            "reference": round(ref, 4),
            "current": round(cur, 4),
            "shift": round(shift, 4),
            "alert": alert,
        }
        if alert:
            alerts.append(intent)
            logger.warning(
                f"INTENT DISTRIBUTION SHIFT: {intent} "
                f"shifted {shift:.1%} from baseline "
                f"(was {ref:.1%}, now {cur:.1%})"
            )

    if not alerts:
        logger.info(
            "Intent distribution check passed — "
            "no significant shifts detected"
        )

    return {
        "shifts": shifts,
        "alerts": alerts,
        "n_alerts": len(alerts),
    }


def check_latency(
    records: list[dict],
    p95_threshold_ms: float = 5000,
) -> dict:
    """
    Checks P50 and P95 latency against thresholds.
    Flags if P95 exceeds threshold.
    """
    df = pd.DataFrame(records)
    p50 = float(df["latency_ms"].quantile(0.50))
    p95 = float(df["latency_ms"].quantile(0.95))

    result = {
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "threshold_ms": p95_threshold_ms,
        "alert": bool(p95 > p95_threshold_ms),
    }

    if result["alert"]:
        logger.warning(
            f"HIGH LATENCY: P95={p95:.0f}ms "
            f"exceeds threshold of {p95_threshold_ms:.0f}ms"
        )
    else:
        logger.info(
            f"Latency check passed: "
            f"P50={p50:.0f}ms P95={p95:.0f}ms"
        )

    return result


def run_monitoring_report() -> dict:
    """
    Runs all monitoring checks on prediction logs.
    Returns full report dict.
    Saves report to data/monitoring/reports/.
    """
    logger.info("=" * 50)
    logger.info("Running SupportMind monitoring report")
    logger.info("=" * 50)

    records = load_prediction_logs()

    if len(records) < 10:
        logger.warning(
            f"Only {len(records)} predictions logged — "
            f"need at least 10 for meaningful monitoring"
        )
        return {
            "error": "insufficient_data",
            "n_records": len(records)
        }

    confidence = check_confidence_degradation(records)
    distribution = check_intent_distribution_shift(records)
    latency = check_latency(records)

    report = {
        "n_predictions": len(records),
        "timestamp": datetime.utcnow().isoformat(),
        "confidence": confidence,
        "distribution": {
            "alerts": distribution.get("alerts", []),
            "n_alerts": distribution.get("n_alerts", 0),
            "shifts": {
                intent: {
                    "reference": float(v["reference"]),
                    "current": float(v["current"]),
                    "shift": float(v["shift"]),
                    "alert": bool(v["alert"]),
                }
                for intent, v in distribution.get(
                    "shifts", {}
                ).items()
            },
        },
        "latency": latency,
    }

    # Save report to disk
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = (
        REPORTS_DIR /
        f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Report saved to {report_path}")

    n_alerts = (
        int(confidence["alert"]) +
        int(latency["alert"]) +
        distribution.get("n_alerts", 0)
    )

    logger.info(f"Total alerts: {n_alerts}")
    logger.info("=" * 50)

    return report


if __name__ == "__main__":
    report = run_monitoring_report()
    print(json.dumps(report, indent=2))