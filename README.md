# SupportMind

Production-grade LLM fine-tuning system for customer support automation.

Fine-tunes Mistral 7B using LoRA to classify customer support ticket
intents and generate accurate, on-brand responses.

## System Components
- Data ingestion and validation pipeline
- LoRA fine-tuning with experiment tracking (MLflow)
- Evaluation harness (F1, BERTScore, LLM-as-judge)
- FastAPI serving layer with input validation
- Monitoring with drift detection (Evidently)

## Pipeline

| Step | Command |
|------|---------|
| Ingest data | `python -m src.data.ingest` |
| Validate data | `python -m src.data.validate` |
| Preprocess | `python -m src.data.preprocess` |
| Train | `python -m src.training.train` |
| Evaluate | `python -m src.evaluation.evaluate` |
| Serve | `uvicorn src.serving.api:app --reload` |

## Quickstart
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m src.data.ingest
```
