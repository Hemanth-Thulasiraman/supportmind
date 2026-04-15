# src/serving/api.py

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from loguru import logger

from src.serving.model import (
    load_model,
    run_inference,
    is_model_loaded,
    ADAPTER_PATH,
    BASE_MODEL_NAME,
)
from src.serving.schemas import (
    TicketRequest,
    TicketResponse,
    HealthResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup. Clean up on shutdown."""
    logger.info("Starting SupportMind API...")
    load_model()
    logger.info("API ready")
    yield
    logger.info("Shutting down SupportMind API")


app = FastAPI(
    title="SupportMind API",
    description=(
        "Customer support ticket intent classification "
        "and response generation using fine-tuned Mistral-7B"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    Returns model status and configuration.
    """
    return HealthResponse(
        status="healthy" if is_model_loaded() else "unhealthy",
        model_loaded=is_model_loaded(),
        base_model=BASE_MODEL_NAME,
        adapter_path=str(ADAPTER_PATH),
    )


@app.post("/predict", response_model=TicketResponse)
def predict(request: TicketRequest):
    """
    Classifies intent and generates response for a support ticket.

    Input: customer support ticket text (10-2000 characters)
    Output: predicted intent, generated response, confidence level
    """
    logger.info(
        f"Predicting for ticket: {request.ticket[:50]}..."
    )

    try:
        result = run_inference(request.ticket)
    except RuntimeError as e:
        logger.error(f"Model not loaded: {e}")
        raise HTTPException(
            status_code=503,
            detail="Model not available. Try again later."
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )

    logger.info(
        f"Predicted intent: {result['intent']} "
        f"(confidence: {result['confidence']})"
    )

    return TicketResponse(**result)