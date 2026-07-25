# SaudiCar AI

Zero to Hero MLOps Bootcamp: build and deploy a production-style machine learning system for predicting used car prices in Saudi Arabia.

## Project Goal

SaudiCar AI is a 30-day, hands-on MLOps bootcamp project. The goal is not only to train a machine learning model, but to understand how a model becomes a real software product: cleaned data, repeatable training, experiment tracking, an API, Docker packaging, CI/CD, cloud deployment, and monitoring.

The first version uses an existing Saudi used car dataset from Kaggle. Later versions may extend the system with new marketplace listings and automatic retraining, only if the data source is legally and technically appropriate.

## What You Will Build

By the end of the bootcamp, this repository should contain:

- A clean Python ML project structure.
- Data cleaning and feature engineering pipelines.
- A trained car price prediction model.
- MLflow experiment tracking.
- A FastAPI prediction service.
- Docker packaging.
- GitHub Actions CI/CD.
- Azure deployment.
- API health checks, logs, and basic monitoring.
- Interview-ready documentation and architecture diagrams.

## Target Architecture

```mermaid
flowchart TD
    A["Saudi Used Car Dataset"] --> B["Data Validation"]
    B --> C["Data Cleaning"]
    C --> D["Feature Engineering"]
    D --> E["Model Training"]
    E --> F["Model Evaluation"]
    F --> G["MLflow Experiments"]
    G --> H["Model Artifact"]
    H --> I["FastAPI Prediction API"]
    I --> J["Docker Image"]
    J --> K["GitHub Actions"]
    K --> L["Azure Deployment"]
    L --> M["Logs and Monitoring"]
    M -. "Optional future phase" .-> N["Scheduled Retraining"]
```

## API Request Flow

```mermaid
sequenceDiagram
    participant User
    participant API as FastAPI Service
    participant Pre as Preprocessing Pipeline
    participant Model as Trained Model
    participant Logs as Monitoring Logs

    User->>API: POST /predict car details
    API->>API: Validate request schema
    API->>Pre: Transform raw input
    Pre->>Model: Send model-ready features
    Model-->>API: Return estimated price
    API->>Logs: Store latency and request metadata
    API-->>User: Prediction response
```

## CI/CD Flow

```mermaid
flowchart LR
    A["Developer Push"] --> B["GitHub Repository"]
    B --> C["GitHub Actions"]
    C --> D["Run Tests"]
    D --> E["Build Docker Image"]
    E --> F["Push Image to Azure Container Registry"]
    F --> G["Deploy to Azure Container Apps"]
    G --> H["Health Check"]
```

## Technology Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| Data | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Experiment Tracking | MLflow |
| API | FastAPI, Pydantic, Uvicorn |
| Packaging | Docker |
| Version Control | Git, GitHub |
| CI/CD | GitHub Actions |
| Cloud | Azure Container Registry, Azure Container Apps or App Service |
| Monitoring | Application logs, health checks, latency/error tracking |

## 30-Day Roadmap

| Week | Theme | Outcome |
| --- | --- | --- |
| Week 1 | ML Foundations | A working baseline model trained on Saudi used car data. |
| Week 2 | ML Engineering | A professional Python project with reusable scripts and a FastAPI service. |
| Week 3 | Core MLOps | Dockerized app, MLflow tracking, model comparison, and basic tests. |
| Week 4 | Production Delivery | CI/CD pipeline, Azure deployment, monitoring, README, and interview demo. |

For the full daily plan, see [docs/BOOTCAMP_PLAN.md](docs/BOOTCAMP_PLAN.md).

For local setup commands, see [docs/DAY_01_SETUP.md](docs/DAY_01_SETUP.md).

For the Python refresh lesson, see [docs/DAY_02_PYTHON_REFRESH.md](docs/DAY_02_PYTHON_REFRESH.md).

For the dataset loading lesson, see [docs/DAY_03_LOAD_DATASET.md](docs/DAY_03_LOAD_DATASET.md).

For dataset details, see [docs/DATASET.md](docs/DATASET.md).

For the exploratory analysis lesson, see [docs/DAY_04_EDA.md](docs/DAY_04_EDA.md).

For the data cleaning lesson, see [docs/DAY_05_DATA_CLEANING.md](docs/DAY_05_DATA_CLEANING.md).

For the feature engineering lesson, see [docs/DAY_06_FEATURE_ENGINEERING.md](docs/DAY_06_FEATURE_ENGINEERING.md).

For the baseline model lesson, see [docs/DAY_07_BASELINE_MODEL.md](docs/DAY_07_BASELINE_MODEL.md).

For the better model lesson, see [docs/DAY_08_BETTER_MODEL.md](docs/DAY_08_BETTER_MODEL.md).

For the repeatable training workflow lesson, see [docs/DAY_09_TRAIN_TEST_WORKFLOW.md](docs/DAY_09_TRAIN_TEST_WORKFLOW.md).

For the first exploratory analysis, see [docs/EDA_SUMMARY.md](docs/EDA_SUMMARY.md).

## Expected Repository Structure

```text
.
|-- data/
|   |-- raw/
|   |-- processed/
|-- docs/
|   |-- BOOTCAMP_PLAN.md
|-- models/
|-- notebooks/
|-- src/
|   |-- saudi_car_ai/
|   |   |-- api/
|   |   |-- data/
|   |   |-- features/
|   |   |-- models/
|   |   |-- monitoring/
|-- tests/
|-- .github/
|   |-- workflows/
|-- Dockerfile
|-- docker-compose.yml
|-- pyproject.toml
|-- README.md
```

## Learning Philosophy

This bootcamp is designed around one rule: every tool must solve a real problem in the project.

- 80 percent coding, 20 percent theory.
- Beginner-friendly MLOps explanations.
- English-first teaching, with Arabic support when needed.
- Small quizzes only.
- No homework.
- Debugging is part of the lesson.
- The student should struggle briefly before receiving help.
- Every week produces a visible project milestone.

## Portfolio Outcome

At the end of the bootcamp, the student should be able to open this repository and explain:

- Where the data came from.
- How the data was cleaned.
- Which features were used.
- Which models were tested.
- Why the final model was selected.
- How MLflow tracks experiments.
- How the FastAPI service returns predictions.
- How Docker packages the application.
- How GitHub Actions automates testing and deployment.
- How the model is deployed on Azure.
- What should be monitored after deployment.
- What future improvements would make the system more production-grade.

## Optional Future Extensions

These are intentionally outside the core 30-day scope:

- Marketplace scraping, if permitted by the source terms.
- Scheduled retraining.
- Model drift detection.
- Prediction history database.
- Authentication.
- Admin dashboard.
- Terraform infrastructure.
- Kubernetes deployment.
- Feature store.

## Project Status

Day 9 repeatable training workflow is now in place. A single command runs the
full pipeline on a fixed train/test split, trains the candidate models (Linear
Regression baseline and Random Forest), selects the best by MAE, and saves both
the chosen `production_model.joblib` and a `training_run.json` manifest that
records the dataset, split, selected model, and metrics. Running it twice gives
the identical result, so training is reproducible. On the real dataset the Random
Forest is selected, cutting average error from about 27,000 SAR to about 12,000
SAR and raising R2 from 0.59 to 0.88.

Model charts (baseline regression line, predicted-vs-actual, error distribution,
and a metrics scoreboard) can be regenerated with
`python -m saudi_car_ai.models.visualize` into `docs/assets/models/`.

Week 1 milestone reached: a single, reproducible command trains and selects a car
price model, saves it, and records exactly how it was produced.
