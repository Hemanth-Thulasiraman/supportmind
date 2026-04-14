# src/evaluation/evaluate.py

import json
import mlflow
import pandas as pd
import torch
from loguru import logger
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.preprocess import format_prompt, get_intent_list
from src.evaluation.metrics import (
    compute_bert_score,
    compute_confusion_matrix,
    compute_intent_f1,
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
    Loads the base model and merges the LoRA adapter.
    Returns model and tokenizer ready for inference.
    """
    logger.info(f"Loading tokenizer from {ADAPTER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading base model: {BASE_MODEL_NAME}")
    from transformers import BitsAndBytesConfig
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
    Returns test_df with added columns:
        predicted_intent, predicted_response, raw_output
    """
    predicted_intents = []
    predicted_responses = []
    raw_outputs = []

    logger.info(f"Running inference on {len(test_df):,} test examples...")

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
                do_sample=False,          # greedy decoding for consistency
                pad_token_id=tokenizer.eos_token_id,
            )

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"].shape[1]
            generated_tokens = output[input_len:]
            raw_text = tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
            raw_outputs.append(raw_text)
            predicted_intents.append(extract_intent_from_output(raw_text))
            predicted_responses.append(extract_response_from_output(raw_text))

        if (i // batch_size) % 10 == 0:
            logger.info(f"Progress: {i:,}/{len(test_df):,} examples")

    test_df = test_df.copy()
    test_df["predicted_intent"] = predicted_intents
    test_df["predicted_response"] = predicted_responses
    test_df["raw_output"] = raw_outputs

    return test_df


def run_evaluation() -> None:
    """
    Main entry point for evaluation pipeline.
    Loads model, runs inference on test set,
    computes all metrics, saves report, logs to MLflow.
    """
    logger.info("=" * 50)
    logger.info("Starting SupportMind evaluation")
    logger.info("=" * 50)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load test data
    test_df = pd.read_parquet(TEST_DATA_PATH)
    logger.info(f"Test set: {len(test_df):,} examples")

    # Get intent list from test data
    intent_list = get_intent_list(test_df)

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Run inference
    results_df = run_inference_on_test_set(
        model, tokenizer, test_df, intent_list
    )

    # Compute metrics
    logger.info("Computing metrics...")

    f1_results = compute_intent_f1(
        y_true=results_df["intent"].tolist(),
        y_pred=results_df["predicted_intent"].tolist(),
    )

    bert_results = compute_bert_score(
        predictions=results_df["predicted_response"].tolist(),
        references=results_df["response"].tolist(),
    )

    cm = compute_confusion_matrix(
        y_true=results_df["intent"].tolist(),
        y_pred=results_df["predicted_intent"].tolist(),
        labels=intent_list,
    )

    # Save results
    results_df.to_parquet(
        RESULTS_DIR / "test_predictions.parquet", index=False
    )
    cm.to_csv(RESULTS_DIR / "confusion_matrix.csv")

    # Save full report as JSON
    report = {
        "macro_f1": f1_results["macro_f1"],
        "weighted_f1": f1_results["weighted_f1"],
        "bertscore_f1": bert_results.get("bertscore_f1", None),
        "bertscore_precision": bert_results.get("bertscore_precision", None),
        "bertscore_recall": bert_results.get("bertscore_recall", None),
        "per_class_f1": f1_results["per_class_f1"],
        "test_size": len(test_df),
    }

    with open(RESULTS_DIR / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Log to MLflow
    mlflow.set_experiment("supportmind-evaluation")
    with mlflow.start_run():
        mlflow.log_metrics({
            "macro_f1": f1_results["macro_f1"],
            "weighted_f1": f1_results["weighted_f1"],
            "bertscore_f1": bert_results.get("bertscore_f1", 0),
        })
        mlflow.log_artifact(str(RESULTS_DIR / "eval_report.json"))
        mlflow.log_artifact(str(RESULTS_DIR / "confusion_matrix.csv"))

    logger.info("=" * 50)
    logger.info("Evaluation complete")
    logger.info(f"Macro F1:    {f1_results['macro_f1']:.4f}")
    logger.info(f"Weighted F1: {f1_results['weighted_f1']:.4f}")
    if bert_results:
        logger.info(
            f"BERTScore F1: {bert_results['bertscore_f1']:.4f}"
        )
    logger.info(f"Report saved to {RESULTS_DIR}")
    logger.info("=" * 50)


if __name__ == "__main__":
    run_evaluation()