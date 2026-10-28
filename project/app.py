"""IndicNER FastAPI Server & Web Application.

High-performance, modern backend serving the interactive HTML5/CSS3/Vanilla JS
two-column workspace for Kannada & Telugu Named Entity Recognition.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.labels import ID2LABEL, LABEL2ID, get_id2label
from src.predict import predict_text
from src.utils import find_project_root, get_device, resolve_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("IndicNER-App")

# Application Paths
PROJ_ROOT = find_project_root()
STATIC_DIR = PROJ_ROOT / "static"
OUTPUTS_DIR = PROJ_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints" / "best_model"

# Initialize FastAPI
app = FastAPI(
    title="IndicNER API",
    description="Cross-Lingual Named Entity Recognition for Kannada & Telugu with MuRIL",
    version="2.0.0",
)

# Enable CORS for flexible integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
MODEL = None
TOKENIZER = None
DEVICE = None
ID2LABEL_MAP = None


def load_model_and_tokenizer():
    """Load model and tokenizer into memory."""
    global MODEL, TOKENIZER, DEVICE, ID2LABEL_MAP
    DEVICE = get_device()
    ID2LABEL_MAP = get_id2label()

    if CHECKPOINT_DIR.exists() and (CHECKPOINT_DIR / "config.json").exists():
        logger.info(f"Loading fine-tuned model checkpoint from: {CHECKPOINT_DIR}")
        model_path = str(CHECKPOINT_DIR)
    else:
        logger.warning(
            f"Checkpoint directory {CHECKPOINT_DIR} not found. "
            "Falling back to base model 'google/muril-base-cased' for inference."
        )
        model_path = "google/muril-base-cased"

    TOKENIZER = AutoTokenizer.from_pretrained(model_path)
    MODEL = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(ID2LABEL_MAP),
        id2label=ID2LABEL_MAP,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    MODEL.to(DEVICE)
    MODEL.eval()
    logger.info(f"Model successfully loaded onto {DEVICE}.")


@app.on_event("startup")
async def startup_event():
    load_model_and_tokenizer()


# Request / Response Schemas
class PredictRequest(BaseModel):
    text: str = Field(..., description="Raw text sentence in Kannada or Telugu")
    min_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Minimum confidence threshold")
    selected_types: List[str] = Field(
        default_factory=lambda: ["PER", "LOC", "ORG"],
        description="List of active entity types to consider",
    )


class SubwordInfo(BaseModel):
    index: int
    word: str
    subwords: List[str]
    count: int
    first_subword: str


class PredictResponse(BaseModel):
    text: str
    language: str
    transfer_mode: str
    tokens: List[str]
    tags: List[str]
    confidences: List[float]
    entities: List[Dict[str, Any]]
    subwords: List[SubwordInfo]


def detect_language(text: str) -> tuple[str, str]:
    """Detect whether text is predominantly Kannada or Telugu."""
    kn_count = 0
    te_count = 0
    for char in text:
        cp = ord(char)
        if 0x0C80 <= cp <= 0x0CFF:
            kn_count += 1
        elif 0x0C00 <= cp <= 0x0C7F:
            te_count += 1

    if kn_count > 0 and kn_count >= te_count:
        return "Kannada", "In-Language Transfer"
    elif te_count > 0:
        return "Telugu", "Zero-Shot Cross-Lingual"
    return "Unknown", "General Script"


# API Endpoints
@app.get("/api/health")
async def health_check():
    """Health status and model metadata."""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": MODEL is not None,
        "checkpoint": str(CHECKPOINT_DIR) if CHECKPOINT_DIR.exists() else "muril-base-cased",
        "labels": ID2LABEL_MAP,
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict_endpoint(payload: PredictRequest):
    """Perform NER extraction on Kannada or Telugu text."""
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model is still initializing.")

    # Run inference
    raw_result = predict_text(
        text=payload.text,
        model=MODEL,
        tokenizer=TOKENIZER,
        device=DEVICE,
        id2label=ID2LABEL_MAP,
    )

    tokens = raw_result["tokens"]
    lang, transfer_mode = detect_language(payload.text)

    # Subword breakdown for inspector
    subwords: List[SubwordInfo] = []
    for idx, word in enumerate(tokens):
        sub_tokens = TOKENIZER.tokenize(word)
        subwords.append(
            SubwordInfo(
                index=idx + 1,
                word=word,
                subwords=sub_tokens,
                count=len(sub_tokens),
                first_subword=sub_tokens[0] if sub_tokens else word,
            )
        )

    return PredictResponse(
        text=raw_result["text"],
        language=lang,
        transfer_mode=transfer_mode,
        tokens=tokens,
        tags=raw_result["tags"],
        confidences=raw_result["confidences"],
        entities=raw_result["entities"],
        subwords=subwords,
    )


@app.get("/api/metrics")
async def get_metrics():
    """Return evaluation metrics JSON if available."""
    metrics_file = OUTPUTS_DIR / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"message": "No evaluation metrics found yet."}


# Mount static assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


@app.get("/")
async def root():
    """Serve the single-page application."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(
            status_code=404,
            content={"message": "Frontend index.html not found in project/static/"},
        )
    return FileResponse(str(index_file))


def main():
    """Launch the IndicNER web server."""
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"\n==================================================================")
    print(f" IndicNER Web Application Running at http://{host}:{port}")
    print(f" Architecture: FastAPI + Custom Modern Responsive HTML5/CSS3/JS UI")
    print(f" Features: Two-Column Layout, Pastel Entity Badges, Subword Inspector")
    print(f"==================================================================\n")
    uvicorn.run("app:app", host=host, port=port, log_level="info", reload=False)


if __name__ == "__main__":
    main()
