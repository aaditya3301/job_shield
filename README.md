# JobShield

[![JobShield CI](https://github.com/aaditya3301/job_shield/actions/workflows/ci.yml/badge.svg)](https://github.com/aaditya3301/job_shield/actions/workflows/ci.yml)

Detect potentially fraudulent job advertisements with TF-IDF and class-weighted Logistic Regression. Includes a trained Kaggle model, FastAPI inference, word-level model explanations, input validation, Docker health checks, and GitHub Actions that test the real model and running container.

## Run locally (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8001
```

Open http://localhost:8001/docs. Expand **POST /predict**, click **Try it out**, paste the example below and click **Execute**.

```json
{
  "title": "Work from home assistant",
  "description": "Earn money immediately. No interview required. Pay a registration fee to start.",
  "requirements": "No experience necessary. Send your bank account details today."
}
```

Predictions are learned from the dataset; a hand-written suspicious example is not guaranteed to be flagged. The API reports the actual model result.

## Run with Docker

```powershell
docker build -t jobshield:latest .
docker run -d --name jobshield-demo -p 127.0.0.1:8000:8000 jobshield:latest
docker ps
python scripts/smoke_test.py
```

Open http://localhost:8000/docs. Use `docker logs jobshield-demo` to inspect startup. Use `docker stop jobshield-demo` when finished, and `docker start jobshield-demo` to resume it. Local Python uses port 8001 so both demonstrations can run together.

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /docs` | Interactive API demo |
| `POST /predict` | Fraud score, risk band, decision and influential words |
| `GET /health` | Model readiness and SHA-256 fingerprint |
| `GET /model-info` | Training metadata, evaluation metrics and thresholds |

Input text joins `title`, `description`, `requirements` in that order, matching training. The model is loaded once during FastAPI startup. Fraud is class 1, with a decision threshold of 0.5. Presentation risk bands are Low below 0.30, Medium from 0.30 to below 0.70, and High at least 0.70; these bands are not separately validated.

Explanations list active TF-IDF features multiplied by their Logistic Regression coefficients. Positive contributions increase the fraud log-odds; negative contributions reduce them. These are model associations, not proof of fraud. Raw probabilities have not been evaluated for calibration. Unrecognized vocabulary is explicitly disclosed.

## Model and evaluation

Dataset: [Kaggle Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction). Training was performed in Kaggle before this repository was assembled.

Metrics below come from metadata inside the user's trained `models/model.joblib`, not from rerunning the dataset locally. Training rows: 14,304. Test rows: 3,576.

| Metric | Test result |
| --- | ---: |
| Accuracy | 97.01% |
| Fraud precision | 64.73% |
| Fraud recall | 83.82% |
| Fraud F1 | 0.7305 |
| ROC-AUC | 0.9764 |
| Average precision | 0.8573 |

Fraud is the minority class, so precision, recall and F1 are more useful than accuracy alone. The reported precision means some genuine advertisements will be flagged; this is a screening aid requiring human review. New scams or job markets may differ from the training data.

The serialized artifact contains `pipeline` and `metadata`. Dependencies pin scikit-learn to the model's training version, 1.7.2. Only load trusted joblib artifacts.

All model artifacts and their export script are in `models/`:

- `models/model.joblib`: fitted vectorizer, classifier and metadata.
- `models/model_parameters.json`: exported hyperparameters, original metrics and artifact fingerprint.
- `models/model_weights.npz`: coefficients, intercept, feature names and IDF weights.
- `models/export_model_info.py`: regenerate those exports from the saved model.

## Tests and CI

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe models/export_model_info.py
```

Every push to `main` runs API tests, builds the Docker image, starts a container, and verifies both `/health` and `/predict`. See [GitHub Actions](https://github.com/aaditya3301/job_shield/actions). This workflow performs continuous integration; it does not deploy a public service or retrain the model.

## Demonstration checklist

1. Show Kaggle training output, classification report and confusion matrix. Save the notebook version and download its `.ipynb` for your submission.
2. Show the saved model and `models/model_parameters.json`.
3. Execute `/predict` in Swagger; explain the score and influential words.
4. Show `docker ps`, container health and a prediction through port 8000.
5. Open this repository's Actions tab and show the successful run.

Runtime patterns follow [FastAPI lifespan documentation](https://fastapi.tiangolo.com/advanced/events/) and the [Docker Python guide](https://docs.docker.com/guides/python/).
