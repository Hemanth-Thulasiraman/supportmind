# src/serving/schemas.py

from pydantic import BaseModel, Field, field_validator


class TicketRequest(BaseModel):
    """Incoming support ticket request."""
    ticket: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Customer support ticket text",
        examples=["I was charged twice for my order last week"],
    )

    @field_validator("ticket")
    @classmethod
    def ticket_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Ticket cannot be empty or whitespace only")
        return v.strip()


class TicketResponse(BaseModel):
    """Predicted intent and generated response."""
    intent: str = Field(
        description="Predicted intent category"
    )
    response: str = Field(
        description="Generated support response"
    )
    confidence: str = Field(
        description="Model confidence: high, medium, or low"
    )
    raw_output: str = Field(
        description="Raw model output for debugging"
    )


class HealthResponse(BaseModel):
    """API health check response."""
    status: str
    model_loaded: bool
    base_model: str
    adapter_path: str