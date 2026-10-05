#!/usr/bin/env bash
# Runs Steps 2-6 after the Airflow ingestion DAG has loaded the raw data.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Building images (first time takes a few minutes)"
docker compose --profile tools --profile serve build

echo "==> Step 2: dbt transformations + data tests"
docker compose run --rm dbt build

echo "==> Step 3: training and comparing models (about 10-20 minutes)"
docker compose run --rm app python -m src.models.train

echo "==> Step 4: SHAP explainability"
docker compose run --rm app python -m src.models.explain

echo "==> Step 6: data drift report"
docker compose run --rm app python -m src.monitoring.drift

echo "==> Step 5: starting the API and MLflow UI"
docker compose up -d api mlflow

echo ""
echo "Done!"
echo "  Model comparison:  reports/model_comparison.md"
echo "  SHAP chart:        reports/shap_summary.png"
echo "  Drift report:      reports/drift_report.md"
echo "  API docs:          http://localhost:8000/docs"
echo "  MLflow UI:         http://localhost:5001"
