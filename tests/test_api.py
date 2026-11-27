from unittest.mock import patch
from fastapi.testclient import TestClient
import pytest
from api import app
from core.sql_engine import load_dataframe, run_sql

client = TestClient(app)

def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_profile_endpoint():
    r = client.get("/profile")
    assert r.status_code == 200
    assert r.json()["rows"] > 0

def test_query_endpoint_with_mocked_llm():
    with patch("api.generate_sql", return_value="SELECT region, SUM(revenue) AS rev FROM data GROUP BY region"), \
         patch("api.summarize_insight", return_value="North leads total revenue."):
        r = client.post("/query", json={"question": "Total revenue by region"})
    assert r.status_code == 200
    body = r.json()
    assert body["repaired"] is False
    assert len(body["rows"]) == 4

def test_blocks_duckdb_file_exfiltration():
    df = load_dataframe("data/sample_sales.csv")
    with pytest.raises(ValueError):
        run_sql(df, "SELECT * FROM read_csv('secret.csv')")