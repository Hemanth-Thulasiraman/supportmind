# src/training/config.py

from pydantic import BaseModel
from pathlib import Path


class LoRAConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "v_proj"]
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class TrainingConfig(BaseModel):
    # Model
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"

    # Data
    train_data_path: Path = Path("data/processed/train.parquet")
    val_data_path: Path = Path("data/processed/val.parquet")

    # Output
    output_dir: Path = Path("models/supportmind-lora")

    # Training
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-4
    warmup_steps: int = 50
    lr_scheduler_type: str = "cosine"

    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 25
    max_seq_length: int = 256

    # Early stopping
    early_stopping_patience: int = 3

    # Quantization
    load_in_4bit: bool = True

    # MLflow
    mlflow_experiment_name: str = "supportmind-finetuning"

    # Reproducibility
    seed: int = 42

    class Config:
        arbitrary_types_allowed = True