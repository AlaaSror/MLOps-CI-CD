from locust import HttpUser, task, between


class ChurnPredictionUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def predict_churn(self):

        payload = {
            "CreditScore": 620,
            "Geography": "France",
            "Gender": "Male",
            "Age": 35,
            "Tenure": 5,
            "Balance": 50000.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 75000.0,
        }

        self.client.post("/predict", json=payload)
