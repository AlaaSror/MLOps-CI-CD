"""
Model loading and prediction logic.

The model must be loaded ONCE at module level, NOT inside the predict function.
"""

import pandas as pd
import joblib
import sklearn.compose._column_transformer as _ct

from app.logger_setup import setup_logging

logger = setup_logging()

logger.info("Starting application ...")

# Compatibility fix for older sklearn pickles
if not hasattr(_ct, "_RemainderColsList"):

    class _RemainderColsList(list):
        pass

    _ct._RemainderColsList = _RemainderColsList

# Load model and transformer
logger.info("Loading churn prediction model ...")
model = joblib.load("data/rf_model.pkl")
logger.info("Model loaded successfully.")

logger.info("Loading transformer ...")
transformer = joblib.load("data/column_transformer.pkl")
logger.info("Transformer loaded successfully.")


def predict_churn(features: pd.DataFrame) -> int:
    """
    Takes a list of feature values and returns a churn prediction (0 or 1).
    """
    # TODO 2: Use model.predict() to get a prediction and return it as an int
    #  Hint: model.predict() expects a 2D array
    logger.debug(f"Received raw features: {features}")

    try:
        logger.info("Running churn prediction...")

        X_transformed = transformer.transform(features)
        prediction = model.predict(X_transformed)

        logger.info(f"Prediction completed successfully: {prediction}")

        return int(prediction[0])
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


if __name__ == "__main__":
    sample = {
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

    logger.info("Testing prediction with sample input.")

    print(f"Input:      {sample}")
    print(f"Prediction: {predict_churn(sample)}")
