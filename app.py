"""JobShield: serve the trained Kaggle text-classification pipeline."""

from contextlib import asynccontextmanager
from hashlib import sha256
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"] if isinstance(artifact, dict) else artifact
    if list(pipeline.classes_) != [0, 1]:
        raise RuntimeError("Expected model classes 0 = genuine and 1 = fraudulent")
    app.state.pipeline = pipeline
    app.state.metadata = artifact.get("metadata", {}) if isinstance(artifact, dict) else {}
    app.state.model_sha256 = sha256(MODEL_PATH.read_bytes()).hexdigest()
    yield


app = FastAPI(
    title="JobShield | Fake Job Posting Detector",
    description=(
        "TF-IDF + Logistic Regression trained on Kaggle job postings. "
        "Submit title, description and requirements to get a model score and "
        "the words influencing it. Scores support review; they do not verify an employer."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class JobPosting(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={"examples": [{
            "title": "Work from home assistant",
            "description": "Earn money immediately. No interview required. Pay a registration fee to start.",
            "requirements": "No experience necessary. Send your bank account details today.",
        }]},
    )
    title: str = Field(default="", max_length=500)
    description: str = Field(min_length=20, max_length=30000)
    requirements: str = Field(default="", max_length=15000)

    @model_validator(mode="after")
    def require_meaningful_text(self):
        if not any(character.isalpha() for character in self.description):
            raise ValueError("Description must contain words")
        return self


class Evidence(BaseModel):
    term: str
    contribution: float


class Prediction(BaseModel):
    prediction: Literal["Genuine", "Potentially fraudulent"]
    fraudulent: int
    fraud_probability: float
    risk_level: Literal["Low", "Medium", "High"]
    decision_threshold: float
    fraud_signals: list[Evidence]
    genuine_signals: list[Evidence]
    recognized_features: int
    note: str


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")


@app.get("/health", tags=["Monitoring"])
def health(request: Request):
    return {"status": "ok", "model_loaded": True, "model_sha256": request.app.state.model_sha256}


@app.get("/model-info", tags=["Monitoring"])
def model_info(request: Request):
    return {
        "metadata": request.app.state.metadata,
        "model_sha256": request.app.state.model_sha256,
        "decision_threshold": 0.5,
        "risk_bands": {"Low": "score < 0.30", "Medium": "0.30 <= score < 0.70", "High": "score >= 0.70"},
        "score_note": "Raw model probability; calibration has not been evaluated.",
    }


@app.post("/predict", response_model=Prediction, tags=["Prediction"])
def predict(job: JobPosting, request: Request):
    pipeline = request.app.state.pipeline
    # Keep exactly the same field order and joining used during Kaggle training.
    text = " ".join([job.title, job.description, job.requirements])
    probability = float(pipeline.predict_proba([text])[0, 1])
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    vector = vectorizer.transform([text]).tocsr()
    names = vectorizer.get_feature_names_out()
    contributions = vector.data * classifier.coef_[0, vector.indices]

    def evidence(positive: bool):
        order = np.argsort(contributions)
        if positive:
            order = order[::-1]
        selected = [i for i in order if (contributions[i] > 0 if positive else contributions[i] < 0)][:5]
        return [Evidence(term=str(names[vector.indices[i]]), contribution=round(float(contributions[i]), 6)) for i in selected]

    fraudulent = int(probability >= 0.5)
    risk = "Low" if probability < 0.3 else "High" if probability >= 0.7 else "Medium"
    note = (
        "Raw model score, not a verified fraud finding. Signals are contributions to the model's log-odds; review the employer independently."
        if vector.nnz else
        "No trained vocabulary was recognized. This score reflects the model baseline and is not informative for this text."
    )
    return Prediction(
        prediction="Potentially fraudulent" if fraudulent else "Genuine",
        fraudulent=fraudulent,
        fraud_probability=probability,
        risk_level=risk,
        decision_threshold=0.5,
        fraud_signals=evidence(True),
        genuine_signals=evidence(False),
        recognized_features=vector.nnz,
        note=note,
    )
