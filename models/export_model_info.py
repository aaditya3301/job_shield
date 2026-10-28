"""Export configuration and learned parameters from the existing trained model."""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

model_dir = Path(__file__).resolve().parent
artifact_path = model_dir / "model.joblib"
artifact = joblib.load(artifact_path)
pipeline = artifact["pipeline"]
vectorizer = pipeline.named_steps["tfidf"]
classifier = pipeline.named_steps["classifier"]
configuration = {
    "metadata": artifact["metadata"],
    "model_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
    "vectorizer_parameters": vectorizer.get_params(),
    "classifier_parameters": classifier.get_params(),
    "vocabulary_size": len(vectorizer.vocabulary_),
    "classes": classifier.classes_.tolist(),
    "decision_threshold": 0.5,
    "metrics_source": "Metadata saved during Kaggle training; not re-evaluated by this export.",
}
(model_dir / "model_parameters.json").write_text(json.dumps(configuration, indent=2, default=str), encoding="utf-8")
np.savez_compressed(model_dir / "model_weights.npz", coefficients=classifier.coef_, intercept=classifier.intercept_, feature_names=vectorizer.get_feature_names_out(), idf=vectorizer.idf_)
print(f"Exported parameters and {len(vectorizer.vocabulary_)} feature weights.")
