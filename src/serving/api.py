# src/serving/api.py

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from loguru import logger
import time
from src.monitoring.logger import log_prediction

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
    Logs every prediction for monitoring.
    """
    logger.info(f"Predicting: {request.ticket[:50]}...")

    start_time = time.time()

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

    latency_ms = (time.time() - start_time) * 1000

    # Log every prediction for monitoring
    log_prediction(
        ticket=request.ticket,
        intent=result["intent"],
        response=result["response"],
        confidence=result["confidence"],
        latency_ms=latency_ms,
    )

    logger.info(
        f"Intent: {result['intent']} "
        f"confidence: {result['confidence']} "
        f"latency: {latency_ms:.0f}ms"
    )

    return TicketResponse(**result)


# Add monitoring endpoint
@app.get("/monitoring/report")
def monitoring_report():
    """
    Runs monitoring checks on prediction logs.
    Returns confidence, distribution, and latency analysis.
    """
    from src.monitoring.drift import run_monitoring_report
    report = run_monitoring_report()
    return report