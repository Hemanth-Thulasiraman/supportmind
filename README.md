# SupportMind — LLM-Powered Customer Support Automation

SupportMind is a production-grade LLM system that classifies customer 
support ticket intent across 27 categories and generates on-brand 
responses using a fine-tuned Mistral-7B model with LoRA adapters.

**Results:** 0.947 macro F1 across 27 intent classes on a held-out 
test set of 2,688 examples.

---

## System Architecture

Raw Ticket (string)
↓
FastAPI Endpoint (/predict)
↓
format_prompt() — shared between training and serving
↓
Mistral-7B + LoRA Adapter (4-bit quantized)
↓
Intent Label + Generated Response
↓
Prediction Logger → Monitoring Reports

---

## Pipeline Components

| Component | File | Description |
|-----------|------|-------------|
| Data Ingestion | `src/data/ingest.py` | Downloads Bitext dataset, validates schema |
| Preprocessing | `src/data/preprocess.py` | Formats prompts, creates stratified splits |
| Training | `src/training/train.py` | LoRA fine-tuning with early stopping |
| Evaluation | `src/evaluation/evaluate.py` | F1, ROUGE, per-class analysis |
| Serving | `src/serving/api.py` | FastAPI with Pydantic validation |
| Monitoring | `src/monitoring/drift.py` | Intent drift, confidence, latency checks |

---

## Results

| Metric | Value |
|--------|-------|
| Macro F1 | 0.947 |
| Weighted F1 | 0.982 |
| ROUGE-1 | 0.510 |
| ROUGE-L | 0.375 |
| Test set size | 2,688 examples |
| Intent classes | 27 |
| Parse failure rate | 0.6% |

**Best performing intents:** change_shipping_address, 
check_cancellation_fee, check_payment_methods, recover_password (F1 = 1.00)

**Hardest intents:** delete_account (0.918), track_refund (0.939) 
— semantically adjacent intents with overlapping language

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/Hemanth-Thulasiraman/supportmind.git
cd supportmind
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run data pipeline
python -m src.data.ingest
python -m src.data.preprocess

# Start API (mock mode without adapter)
uvicorn src.serving.api:app --reload --port 8000
```

---

## API Usage

**Health check:**
```bash
curl http://localhost:8000/health
```

**Predict intent and generate response:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticket": "I was charged twice for my order last week"}'
```

**Response:**
```json
{
  "intent": "payment_issue",
  "response": "We apologize for the duplicate charge...",
  "confidence": "high",
  "raw_output": "payment_issue\nResponse: ..."
}
```

**Monitoring report:**
```bash
curl http://localhost:8000/monitoring/report
```

---

## Training

Training requires a GPU with at least 16GB VRAM. 
Tested on NVIDIA A100 80GB via Google Colab.

```bash
# Full training pipeline
python -m src.training.train

# Evaluation on test set
python -m src.evaluation.evaluate
```

**Training configuration:**
- Base model: `mistralai/Mistral-7B-Instruct-v0.2`
- LoRA rank: 64, alpha: 128
- Target modules: q_proj, v_proj, k_proj, o_proj
- Quantization: 4-bit NF4
- Early stopping patience: 3 evaluations
- Best checkpoint: step 200, eval_loss = 0.326

---

## Docker

```bash
# Build
docker build -t supportmind .

# Run
docker run -p 8000:8000 supportmind

# Health check
curl http://localhost:8000/health
```

---

## Project Structure
supportmind/
├── src/
│   ├── data/
│   │   ├── ingest.py        # HuggingFace dataset download + validation
│   │   ├── validate.py      # Schema, null, distribution checks
│   │   └── preprocess.py    # Prompt formatting + stratified splits
│   ├── training/
│   │   ├── config.py        # Pydantic hyperparameter config
│   │   └── train.py         # LoRA fine-tuning with SFTTrainer
│   ├── evaluation/
│   │   ├── metrics.py       # F1, ROUGE, confusion matrix, parsers
│   │   └── evaluate.py      # Full test set evaluation pipeline
│   ├── serving/
│   │   ├── api.py           # FastAPI application
│   │   ├── model.py         # Model loader and inference
│   │   └── schemas.py       # Pydantic request/response models
│   └── monitoring/
│       ├── logger.py        # Prediction logging to JSONL
│       └── drift.py         # Distribution shift + latency monitoring
├── configs/                 # Training configuration files
├── data/
│   ├── raw/                 # Raw dataset (gitignored)
│   ├── processed/           # Train/val/test splits (gitignored)
│   └── validation_reports/  # Evaluation reports
├── models/                  # LoRA adapter weights (gitignored)
├── Dockerfile
├── requirements.txt
└── README.md

---

## Key Engineering Decisions

**Why LoRA over full fine-tuning?**
LoRA trains only 0.74% of model parameters (54M of 7.3B), reducing 
GPU memory from 28GB to 5GB with 4-bit quantization. This makes 
fine-tuning feasible on a single GPU without sacrificing performance.

**Why a shared format_prompt() function?**
Training and serving import the same `format_prompt()` from 
`src/data/preprocess.py`. Any prompt format change updates both 
simultaneously, eliminating training-serving skew — one of the most 
common silent failure modes in production ML.

**Why early stopping?**
Best checkpoint was at step 200 with eval_loss = 0.326. Without early 
stopping, training would have continued to step 2016 and overfit — 
eval_loss was already rising by step 300.

---

## Monitoring

The `/monitoring/report` endpoint runs three checks:

| Check | Signal | Alert Threshold |
|-------|--------|-----------------|
| Confidence degradation | % low confidence predictions | > 10% |
| Intent distribution shift | KL divergence from training baseline | > 5pp per intent |
| Latency | P95 response time | > 5000ms |

---

## Retrospective

**What worked well:**
- LoRA fine-tuning converged fast — 0.947 F1 in under an hour on A100
- Modular pipeline design made debugging straightforward
- Early stopping correctly identified step 200 as the optimal checkpoint
- Pydantic validation caught malformed inputs before they reached the model

**What I would do differently:**
- Pin exact library versions from the start — TRL breaking changes cost hours
- Save inference predictions to disk immediately — lost one full eval run
- Use left padding from the start — discovered the right-padding issue late

**Version 2 improvements:**
- LLM-as-judge evaluation for response quality scoring
- A/B testing framework for comparing model versions
- Retraining pipeline triggered by monitoring alerts
- BERTScore evaluation with stable library version
- HuggingFace Hub model registry integration