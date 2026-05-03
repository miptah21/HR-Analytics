"""
Load Testing Script — HR Attrition Intelligence API (G-14)

Uses Locust to simulate concurrent prediction requests and measure
API performance under load. Target: 100 concurrent users, < 500ms P95.

Usage:
    # Install: uv pip install locust
    # Run:     locust -f tests/load_test.py --host=http://localhost:8000
    # Headless: locust -f tests/load_test.py --host=http://localhost:8000 \
    #           --users 100 --spawn-rate 10 --run-time 60s --headless
"""
import json
import random

from locust import HttpUser, between, task


# Sample employee payloads for realistic load testing
SAMPLE_EMPLOYEES = [
    {
        "EmployeeID": f"LOAD-{i:04d}",
        "Age": random.randint(22, 60),
        "JobRole": random.choice([
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources",
        ]),
        "JobLevel": random.randint(1, 5),
        "MonthlyIncome": random.randint(2000, 20000),
        "PercentSalaryHike": random.randint(11, 25),
        "OverTime": random.choice(["Yes", "No"]),
        "DistanceFromHome": random.randint(1, 29),
        "WorkLifeBalance": random.randint(1, 4),
        "YearsAtCompany": random.randint(0, 30),
        "YearsInCurrentRole": random.randint(0, 15),
        "YearsSinceLastPromotion": random.randint(0, 15),
        "YearsWithCurrManager": random.randint(0, 15),
        "TotalWorkingYears": random.randint(1, 40),
        "JobSatisfaction": random.randint(1, 4),
        "EnvironmentSatisfaction": random.randint(1, 4),
        "RelationshipSatisfaction": random.randint(1, 4),
        "JobInvolvement": random.randint(1, 4),
        "BusinessTravel": random.choice([
            "Travel_Rarely", "Travel_Frequently", "Non-Travel",
        ]),
    }
    for i in range(50)
]


class HRAnalyticsUser(HttpUser):
    """Simulates an HR professional using the API."""

    wait_time = between(1, 3)
    headers = {
        "Content-Type": "application/json",
        # In production, set via LOCUST_API_KEY env var
        "X-API-Key": "",
    }

    @task(5)
    def predict_attrition(self):
        """POST /v1/predict — most common operation."""
        payload = random.choice(SAMPLE_EMPLOYEES)
        self.client.post(
            "/v1/predict",
            json=payload,
            headers=self.headers,
            name="/v1/predict",
        )

    @task(3)
    def get_dashboard_summary(self):
        """GET /v1/dashboard/summary — executive overview."""
        self.client.get(
            "/v1/dashboard/summary",
            headers=self.headers,
            name="/v1/dashboard/summary",
        )

    @task(2)
    def get_dashboard_employees(self):
        """GET /v1/dashboard/employees — full employee list."""
        self.client.get(
            "/v1/dashboard/employees",
            headers=self.headers,
            name="/v1/dashboard/employees",
        )

    @task(2)
    def get_drift_status(self):
        """GET /v1/dashboard/drift — monitoring check."""
        self.client.get(
            "/v1/dashboard/drift",
            headers=self.headers,
            name="/v1/dashboard/drift",
        )

    @task(1)
    def health_check(self):
        """GET /v1/health — lightweight ping."""
        self.client.get("/v1/health", name="/v1/health")

    @task(1)
    def get_trends(self):
        """GET /v1/dashboard/trends — trend analysis."""
        self.client.get(
            "/v1/dashboard/trends?days=30",
            headers=self.headers,
            name="/v1/dashboard/trends",
        )


class HRAuditorUser(HttpUser):
    """Simulates an auditor reviewing logs (lower frequency)."""

    wait_time = between(5, 10)
    weight = 1  # Lower weight = fewer auditor users
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": "",
    }

    @task(3)
    def get_audit_logs(self):
        """GET /v1/audit-logs — audit trail review."""
        self.client.get(
            "/v1/audit-logs?page=1&page_size=20",
            headers=self.headers,
            name="/v1/audit-logs",
        )

    @task(1)
    def get_exclusions(self):
        """GET /v1/gdpr/exclusions — compliance check."""
        self.client.get(
            "/v1/gdpr/exclusions",
            headers=self.headers,
            name="/v1/gdpr/exclusions",
        )
