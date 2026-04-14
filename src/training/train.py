# src/training/train.py

import mlflow
import pandas as pd
import torch
from datasets import Dataset
from loguru import logger
from pathlib import Path
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTTrainer, SFTConfig

from src.training.config import TrainingConfig, LoRAConfig


def load_tokenizer(config: TrainingConfig):
    """
    Loads and configures the tokenizer.
    Sets pad token to eos token — required for causal LM training.
    """
    logger.info(f"Loading tokenizer: {config.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(config: TrainingConfig):
    """
    Loads base model with 4-bit quantization for GPU,
    or full precision for CPU fallback.
    """
    logger.info(f"Loading base model: {config.base_model_name}")

    if torch.cuda.is_available() and config.load_in_4bit:
        logger.info("CUDA detected — loading with 4-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        logger.warning("No CUDA — loading in full precision on CPU")
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            dtype=torch.float32,
        )

    return model


def apply_lora(model, lora_config: LoRAConfig):
    """
    Injects LoRA adapter into the base model.
    Only adapter weights will be trained.
    """
    logger.info(
        f"Applying LoRA: rank={lora_config.r}, "
        f"alpha={lora_config.lora_alpha}"
    )

    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def load_datasets(config: TrainingConfig):
    """
    Loads train and validation parquet files.
    Renames prompt column to text for SFTTrainer compatibility.
    Converts to HuggingFace Dataset format.
    """
    logger.info("Loading train and validation datasets...")

    train_df = pd.read_parquet(config.train_data_path)
    val_df = pd.read_parquet(config.val_data_path)

    # Rename for SFTTrainer compatibility
    train_df = train_df.rename(columns={"prompt": "text"})
    val_df = val_df.rename(columns={"prompt": "text"})

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    logger.info(f"Train: {len(train_dataset):,} examples")
    logger.info(f"Val:   {len(val_dataset):,} examples")

    return train_dataset, val_dataset


def run_training(
    config: TrainingConfig = None,
    lora_config: LoRAConfig = None,
) -> None:
    """
    Main entry point for training pipeline.
    Loads model, applies LoRA, trains, logs to MLflow,
    saves adapter to output_dir.
    """
    if config is None:
        config = TrainingConfig()
    if lora_config is None:
        lora_config = LoRAConfig()

    logger.info("=" * 50)
    logger.info("Starting SupportMind fine-tuning")
    logger.info("=" * 50)

    mlflow.set_experiment(config.mlflow_experiment_name)

    with mlflow.start_run():

        # Log all hyperparameters
        mlflow.log_params({
            "base_model": config.base_model_name,
            "epochs": config.num_train_epochs,
            "learning_rate": config.learning_rate,
            "batch_size": config.per_device_train_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": (
                config.per_device_train_batch_size *
                config.gradient_accumulation_steps
            ),
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "max_seq_length": config.max_seq_length,
            "seed": config.seed,
        })

        # Load all components
        tokenizer = load_tokenizer(config)
        model = load_base_model(config)
        model = apply_lora(model, lora_config)
        train_dataset, val_dataset = load_datasets(config)

        # SFTTrainer configuration
        sft_config = SFTConfig(
            output_dir=str(config.output_dir),
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            warmup_steps=config.warmup_steps,
            lr_scheduler_type=config.lr_scheduler_type,
            eval_strategy=config.eval_strategy,
            eval_steps=config.eval_steps,
            save_steps=config.save_steps,
            logging_steps=config.logging_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=config.seed,
            dataset_text_field="text",
            max_length=config.max_seq_length,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=sft_config,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=config.early_stopping_patience
                )
            ],
        )

        # Train
        logger.info("Starting training...")
        train_result = trainer.train()

        # Log final metrics
        mlflow.log_metrics({
            "train_loss": train_result.training_loss,
            "train_runtime_seconds": (
                train_result.metrics["train_runtime"]
            ),
            "train_samples_per_second": (
                train_result.metrics["train_samples_per_second"]
            ),
        })

        # Save adapter and tokenizer
        logger.info(f"Saving LoRA adapter to {config.output_dir}")
        config.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(config.output_dir))
        tokenizer.save_pretrained(str(config.output_dir))

        # Log adapter as MLflow artifact
        mlflow.log_artifacts(
            str(config.output_dir),
            artifact_path="lora-adapter"
        )

        logger.info("Training complete")
        logger.info(f"Adapter saved to: {config.output_dir}")


if __name__ == "__main__":
    run_training()