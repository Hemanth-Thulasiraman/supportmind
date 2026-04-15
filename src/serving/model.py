# src/serving/model.py

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
    extract_intent_from_output,
    extract_response_from_output,
)

ADAPTER_PATH = Path("models/supportmind-lora")
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
MAX_NEW_TOKENS = 128
MOCK_MODE = not (ADAPTER_PATH / "adapter_model.safetensors").exists()

# Global model state — loaded once at startup
_intent_list: list[str] = []
_model = None
_tokenizer = None


def load_intent_list() -> list[str]:
    """
    Loads intent list from processed training data.
    Falls back to hardcoded list if data not available.
    """
    try:
        import pandas as pd
        df = pd.read_parquet("data/processed/train.parquet")
        intents = get_intent_list(df)
        logger.info(f"Loaded {len(intents)} intents from training data")
        return intents
    except Exception as e:
        logger.warning(f"Could not load intents from data: {e}")
        logger.warning("Using fallback hardcoded intent list")
        return [
            "cancel_order", "change_order", "change_shipping_address",
            "check_cancellation_fee", "check_invoice",
            "check_payment_methods", "check_refund_policy",
            "complaint", "contact_customer_service",
            "contact_human_agent", "create_account", "delete_account",
            "delivery_options", "delivery_period", "edit_account",
            "get_invoice", "get_refund", "newsletter_subscription",
            "payment_issue", "place_order", "recover_password",
            "registration_problems", "review",
            "set_up_shipping_address", "switch_account",
            "track_order", "track_refund",
        ]


def load_model() -> None:
    """
    Loads base model and LoRA adapter into memory.
    Called once at API startup — not on every request.
    Falls back to mock mode if adapter not found locally.
    """
    global _model, _tokenizer, _intent_list

    _intent_list = load_intent_list()

    if MOCK_MODE:
        logger.warning(
            "Adapter not found at models/supportmind-lora — "
            "running in MOCK MODE for local testing"
        )
        _model = "mock"
        _tokenizer = "mock"
        return

    logger.info("Loading tokenizer...")
    _tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = "left"

    logger.info(f"Loading base model: {BASE_MODEL_NAME}")

    if torch.cuda.is_available():
        logger.info("CUDA detected — using 4-bit quantization")
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
    else:
        logger.warning("No CUDA — loading on CPU")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            dtype=torch.float32,
        )

    logger.info(f"Loading LoRA adapter from {ADAPTER_PATH}")
    _model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
    _model.eval()
    logger.info("Model ready for inference")


def is_model_loaded() -> bool:
    """Returns True if model and tokenizer are loaded."""
    return _model is not None and _tokenizer is not None


def run_inference(ticket: str) -> dict:
    """
    Runs inference on a single support ticket.
    Returns intent, response, confidence, and raw model output.
    """
    if not is_model_loaded():
        raise RuntimeError(
            "Model not loaded. Call load_model() first."
        )

    # Mock mode for local testing without GPU or adapter
    if MOCK_MODE:
        logger.warning("MOCK MODE: returning fake prediction")
        return {
            "intent": "payment_issue",
            "response": (
                "We apologize for the duplicate charge on your account. "
                "Please allow 3-5 business days for the refund to process."
            ),
            "confidence": "high",
            "raw_output": (
                "payment_issue\nResponse: We apologize for the "
                "duplicate charge on your account."
            ),
        }

    # Format prompt using shared formatter — same function as training
    # This is critical: same format_prompt = no training-serving skew
    prompt = format_prompt(
        instruction=ticket,
        intent_list=_intent_list,
    )

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(_model.device)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Decode only the generated tokens, not the input prompt
    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]
    raw_text = _tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )

    intent = extract_intent_from_output(raw_text, _intent_list)
    response = extract_response_from_output(raw_text)

    # Confidence based on whether intent parsed cleanly
    if intent == "unknown":
        confidence = "low"
    elif intent in _intent_list:
        confidence = "high"
    else:
        confidence = "medium"

    return {
        "intent": intent,
        "response": response,
        "confidence": confidence,
        "raw_output": raw_text,
    }