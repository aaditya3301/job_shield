# JobShield

[![JobShield CI](https://github.com/aaditya3301/job_shield/actions/workflows/ci.yml/badge.svg)](https://github.com/aaditya3301/job_shield/actions/workflows/ci.yml)

JobShield uses machine learning to flag potentially fraudulent job postings. Training and testing are completed in a Kaggle notebook; FastAPI serves predictions locally, and Docker packages the API and trained model.

## Dataset

[Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) contains 17,880 job advertisements. The target column is `fraudulent`: `0` means genuine and `1` means fraudulent. Job title, description, and requirements are combined as model input. The dataset is imbalanced, with 866 fraudulent examples.

## Model

The Kaggle notebook trains a scikit-learn pipeline consisting of TF-IDF text vectorization and class-weighted Logistic Regression. It uses an 80/20 train/test split: 14,304 training records and 3,576 test records.

| Component | Parameters |
| --- | --- |
| TF-IDF | English stop words, unigrams and bigrams, `min_df=2`, `max_features=12000`, `sublinear_tf=True` |
| Logistic Regression | `class_weight="balanced"`, `C=1.0`, `penalty="l2"`, `solver="liblinear"`, `max_iter=700`, `random_state=42` |
| Prediction | Fraud class `1`; decision threshold `0.5` |

Test-set results: 97.01% overall accuracy; for the fraudulent class, 64.73% precision, 83.82% recall, and 0.7305 F1. Because the dataset is imbalanced, precision and recall help explain performance on fraudulent postings more clearly than accuracy alone.

The trained pipeline and metadata are saved in `models/model.joblib`. `models/model_parameters.json` records configuration and evaluation results; `models/model_weights.npz` contains learned weights and TF-IDF features. `models/export_model_info.py` regenerates these exports.

## Pipeline

1. Select the Kaggle fake job posting dataset.
2. Train and evaluate the model in a Kaggle notebook.
3. Save the trained pipeline and its parameters in `models/`.
4. Serve predictions with FastAPI from `app.py` in VS Code. The API returns a fraud score, risk level, and influential words.
5. Build and run the API in Docker.
6. Store the project in the [JobShield GitHub repository](https://github.com/aaditya3301/job_shield).
7. GitHub Actions runs on pushes and pull requests. It runs the API tests, builds the Docker image, and checks the health and prediction endpoints.
8. See the [GitHub Actions workflow](https://github.com/aaditya3301/job_shield/actions) for run status.

## Run locally

Requires Python 3.11 and Docker.

In PowerShell, install dependencies and start the API:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8001
```

Open http://localhost:8001/docs and try **POST /predict**.

To build and run the Docker version:

```powershell
docker build -t jobshield:latest .
docker run --rm -p 8000:8000 jobshield:latest
```

Open http://localhost:8000/docs. The API also provides `GET /health` and `GET /model-info`.
