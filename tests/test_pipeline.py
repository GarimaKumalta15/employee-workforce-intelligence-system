"""Run with:  pytest -q"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generate_dataset import generate          # noqa: E402
from ml.features import clean, BASE                  # noqa: E402


@pytest.fixture(scope="module")
def client():
    from app import app
    return app.test_client()


def test_dataset_has_mess_and_cleaning_fixes_it():
    raw = generate(n=300, seed=1)
    assert raw.duplicated("employee_id").sum() == 8
    assert raw.isna().sum().sum() > 0
    d = clean(raw)
    assert d.employee_id.is_unique and len(d) == 300
    assert set(["attrition_flag", "promotion_flag", "training_flag", "overtime_flag"]) <= set(d.columns)


def test_overview_and_list(client):
    k = client.get("/api/overview").get_json()["kpis"]
    assert k["total"] == 1500 and 0.05 < k["attrition_rate"] < 0.35
    r = client.get("/api/employees?risk=High&per_page=5").get_json()
    assert r["total"] > 0 and all(e["risk_level"] == "High" for e in r["rows"])


def test_search_and_employee_report(client):
    r = client.get("/api/employees?q=EMP0007").get_json()
    assert r["rows"][0]["employee_id"] == "EMP0007"
    rep = client.get("/api/employee/emp0007").get_json()
    p = rep["predictions"]
    assert 0 <= p["attrition_prob"] <= 1 and 1 <= p["pred_performance"] <= 5
    assert p["risk_level"] in {"Low", "Medium", "High"} and rep["recommendations"]
    assert client.get("/api/employee/NOPE").status_code == 404


def test_whatif_risky_profile_scores_higher_than_happy_profile(client):
    risky = {"age": 24, "overtime": "Yes", "job_satisfaction": 1, "work_life_balance": 1,
             "environment_satisfaction": 1, "years_since_last_promotion": 6, "monthly_income": 30000,
             "department": "Sales", "job_role": "Sales Executive", "business_travel": "Travel_Frequently"}
    happy = {"age": 45, "overtime": "No", "job_satisfaction": 4, "work_life_balance": 4,
             "environment_satisfaction": 4, "years_since_last_promotion": 0, "monthly_income": 90000,
             "department": "Sales", "job_role": "Sales Manager", "years_at_company": 12}
    a = client.post("/api/predict", json=risky).get_json()["predictions"]
    b = client.post("/api/predict", json=happy).get_json()["predictions"]
    assert a["attrition_prob"] > b["attrition_prob"] + 0.2
    assert client.post("/api/predict", json={}).status_code == 400


def test_report_models_and_anomalies(client):
    assert len(client.get("/api/report").get_json()["findings"]) >= 5
    m = client.get("/api/model-info").get_json()
    assert m["metrics"]["attr"]["roc_auc"] > 0.75 and m["metrics"]["perf"]["r2"] > 0.5
    assert len(client.get("/api/anomalies").get_json()) > 0
    assert client.get("/").status_code == 200
