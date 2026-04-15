# src/evaluation/evaluate.py

import json
import mlflow
import pandas as pd
import torch
from loguru import logger
from pathlib import Path
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from src.data.preprocess import format_prompt, get_intent_list
from src.evaluation.metrics import (
    compute_confusion_matrix,
    compute_intent_f1,
    compute_rouge,
    extract_intent_from_output,
    extract_response_from_output,
)

TEST_DATA_PATH = Path("data/processed/test.parquet")
ADAPTER_PATH = Path("models/supportmind-lora")
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
RESULTS_DIR = Path("data/validation_reports")
MAX_NEW_TOKENS = 128


def load_model_and_tokenizer():
    """
    Loads base model and merges LoRA adapter.
    Sets padding_side=left for correct decoder generation.
    """
    logger.info(f"Loading tokenizer from {ADAPTER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info(f"Loading base model: {BASE_MODEL_NAME}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    logger.info(f"Loading LoRA adapter from {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
    model.eval()
    logger.info("Model loaded and ready for inference")
    return model, tokenizer


def run_inference_on_test_set(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    intent_list: list[str],
    batch_size: int = 8,
) -> pd.DataFrame:
    """
    Runs inference on every example in the test set.
    Returns test_df with predicted_intent, predicted_response,
    and raw_output columns added.
    """
    predicted_intents = []
    predicted_responses = []
    raw_outputs = []

    logger.info(
        f"Running inference on {len(test_df):,} test examples..."
    )

    for i in range(0, len(test_df), batch_size):
        batch = test_df.iloc[i:i + batch_size]

        prompts = [
            format_prompt(
                instruction=row["instruction"],
                intent_list=intent_list,
            )
            for _, row in batch.iterrows()
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"].shape[1]
            generated_tokens = output[input_len:]
            raw_text = tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
            raw_outputs.append(raw_text)
            predicted_intents.append(
                extract_intent_from_output(raw_text, intent_list)
            )
            predicted_responses.append(
                extract_response_from_output(raw_text)
            )

        if (i // batch_size) % 10 == 0:
            logger.info(
                f"Progress: {i:,}/{len(test_df):,} examples"
            )

    result_df = test_df.copy()
    result_df["predicted_intent"] = predicted_intents
    result_df["predicted_response"] = predicted_responses
    result_df["raw_output"] = raw_outputs
    return result_df


def run_evaluation(dev_mode: bool = False) -> None:
    """
    Main entry point for evaluation pipeline.
    Loads predictions from disk if they exist — skips inference.
    Runs all metrics and saves report.
    """
    logger.info("=" * 50)
    logger.info("Starting SupportMind evaluation")
    logger.info("=" * 50)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = RESULTS_DIR / "test_predictions.parquet"

    # Skip inference if predictions already saved
    if predictions_path.exists():
        logger.info("Found existing predictions — loading from disk")
        results_df = pd.read_parquet(predictions_path)
        intent_list = sorted(results_df["intent"].unique().tolist())
    else:
        test_df = pd.read_parquet(TEST_DATA_PATH)

        if dev_mode:
            test_df = test_df.head(50)
            logger.warning("DEV MODE: running on 50 examples only")

        logger.info(f"Test set: {len(test_df):,} examples")
        intent_list = get_intent_list(test_df)
        model, tokenizer = load_model_and_tokenizer()

        results_df = run_inference_on_test_set(
            model, tokenizer, test_df, intent_list
        )

        # Save immediately before any metric computation
        results_df.to_parquet(predictions_path, index=False)
        logger.info(f"Predictions saved to {predictions_path}")

    # Show sample outputs to verify parsing
    logger.info("=== SAMPLE RAW OUTPUTS (first 5) ===")
    for i in range(min(5, len(results_df))):
        logger.info(
            f"True:      {results_df['intent'].iloc[i]}"
        )
        logger.info(
            f"Predicted: {results_df['predicted_intent'].iloc[i]}"
        )
        logger.info(
            f"Raw:       "
            f"{repr(results_df['raw_output'].iloc[i][:200])}"
        )
        logger.info("---")

    # Parse failure check
    unknown_count = (
        results_df["predicted_intent"] == "unknown"
    ).sum()
    unknown_pct = unknown_count / len(results_df) * 100
    logger.info(
        f"Parse failures: {unknown_count}/{len(results_df)} "
        f"({unknown_pct:.1f}%)"
    )

    if unknown_pct > 50:
        logger.error(
            "Over 50% parse failures. "
            "Check raw outputs above before proceeding."
        )
        return

    # Compute metrics
    logger.info("Computing metrics...")

    f1_results = compute_intent_f1(
        y_true=results_df["intent"].tolist(),
        y_pred=results_df["predicted_intent"].tolist(),
    )

    cm = compute_confusion_matrix(
        y_true=results_df["intent"].tolist(),
        y_pred=results_df["predicted_intent"].tolist(),
        labels=intent_list,
    )
    cm.to_csv(RESULTS_DIR / "confusion_matrix.csv")

    rouge_results = compute_rouge(
        predictions=results_df["predicted_response"].tolist(),
        references=results_df["response"].tolist(),
    )

    # Save full report
    report = {
        "macro_f1": f1_results["macro_f1"],
        "weighted_f1": f1_results["weighted_f1"],
        "rouge1": rouge_results.get("rouge1", None),
        "rougeL": rouge_results.get("rougeL", None),
        "per_class_f1": f1_results["per_class_f1"],
        "test_size": len(results_df),
        "parse_failure_rate": unknown_pct,
    }

    with open(RESULTS_DIR / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Log to MLflow
    mlflow.set_experiment("supportmind-evaluation")
    with mlflow.start_run():
        mlflow.log_metrics({
            "macro_f1": f1_results["macro_f1"],
            "weighted_f1": f1_results["weighted_f1"],
            "rouge1": rouge_results.get("rouge1", 0),
            "rougeL": rouge_results.get("rougeL", 0),
        })
        mlflow.log_artifact(
            str(RESULTS_DIR / "eval_report.json")
        )
        mlflow.log_artifact(
            str(RESULTS_DIR / "confusion_matrix.csv")
        )

    logger.info("=" * 50)
    logger.info("Evaluation complete")
    logger.info(f"Macro F1:    {f1_results['macro_f1']:.4f}")
    logger.info(f"Weighted F1: {f1_results['weighted_f1']:.4f}")
    if rouge_results:
        logger.info(
            f"ROUGE-1:     {rouge_results.get('rouge1', 0):.4f}"
        )
        logger.info(
            f"ROUGE-L:     {rouge_results.get('rougeL', 0):.4f}"
        )
    logger.info(f"Report saved to {RESULTS_DIR}")
    logger.info("=" * 50)


if __name__ == "__main__":
    import sys
    dev_mode = "--dev" in sys.argv
    run_evaluation(dev_mode=dev_mode)