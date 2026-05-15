"""
Tests for the Churn Prediction API.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov=main --cov-report=term-missing
"""

import pandas as pd
import pytest
from main import app
from litestar.testing import TestClient
# ---------------------------------------------------------------------------
# Function Tests
# ---------------------------------------------------------------------------


# TODO 1: Write a test that calls predict_churn() directly with sample features
#         and asserts the result is 0 or 1
#         Hint: import predict_churn from app.model_utils
def test_predict_churn():
    from app.model_utils import predict_churn

    # Sample input features (replace with actual feature names and values)
    sample_features = {
        "CreditScore": 619,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 101348.9,
    }

    result = predict_churn(pd.DataFrame([sample_features]))
    assert result in [0, 1], f"Expected 0 or 1, got {result}"


# TODO 2 (bonus): Write another function test with edge-case inputs
def test_predict_churn_edge_cases():
    from app.model_utils import predict_churn

    # Edge case input features (e.g., missing values, extreme values)
    edge_case_features = {
        "CreditScore": 0,
        "Geography": "Unknown",  # Wrong category
        "Gender": "Other",  # Unseen category
        "Age": 120,  # Extreme high
        "Tenure": 0,
        "Balance": -100.0,  # Invalid negative balance
        "NumOfProducts": 0,
        "HasCrCard": 0,
        "IsActiveMember": 0,
        "EstimatedSalary": 101348.0,
    }

    try:
        result = predict_churn(pd.DataFrame([edge_case_features]))
        assert result in (0, 1), f"Expected 0 or 1, got {result}"
        print("passed")

    except Exception as e:
        pytest.fail(f"predict_churn raised an exception due to edge case inputs: {e}")
        print("failed")


# ---------------------------------------------------------------------------
# Endpoint Tests
# ---------------------------------------------------------------------------


# TODO 3: Write a test that POSTs to /predict with valid JSON
#         and checks the status code and response body
#         Hint: Litestar POST returns 201, not 200
#         Hint: use `with TestClient(app=app) as client:
def test_predict_post():
    payload = {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0,
    }

    with TestClient(app=app) as client:
        response = client.post("/predict", json=payload)

        assert response.status_code in (200, 201)
        body = response.json()
        assert "prediction" in body


# TODO 4: Write a test for GET /health
def test_get_health():
    with TestClient(app=app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


# TODO 5: Write a test for GET /
def test_get_root():
    with TestClient(app=app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcom to the Churn Prediction Model"}


# TODO 6 (bonus): Test that invalid input returns status 400
def test_post_invalid_input():
    with TestClient(app=app) as client:
        response = client.post(
            "/predict",
            json={
                "CreditScore": "invalid",  # Should be int
                "Geography": "France",
                "Gender": "Female",
                "Age": 42,
                "Tenure": 2,
                "Balance": 0.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 101348.9,
            },
        )
        assert response.status_code == 400
