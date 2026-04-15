# src/monitoring/logger.py

import json
from datetime import datetime
from pathlib import Path
from loguru import logger

LOG_DIR = Path("data/monitoring")


def log_prediction(
    ticket: str,
    intent: str,
    response: str,
    confidence: str,
    latency_ms: float,
) -> None:
    """
    Logs a single prediction to a JSONL file.
    One JSON record per line for easy streaming reads.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "ticket": ticket,
        "ticket_length": len(ticket),
        "predicted_intent": intent,
        "confidence": confidence,
        "response_length": len(response),
        "latency_ms": round(latency_ms, 2),
    }

    log_file = LOG_DIR / "predictions.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(record) + "\n")

    logger.debug(
        f"Logged prediction: intent={intent} "
        f"confidence={confidence} "
        f"latency={latency_ms:.0f}ms"
    )


def load_prediction_logs() -> list[dict]:
    """
    Loads all prediction logs from disk.
    Returns list of prediction records.
    """
    log_file = LOG_DIR / "predictions.jsonl"

    if not log_file.exists():
        logger.warning("No prediction logs found")
        return []

    records = []
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info(f"Loaded {len(records):,} prediction records")
    return records