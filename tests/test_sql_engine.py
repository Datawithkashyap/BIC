import pytest
from core.sql_engine import load_dataframe, run_sql, run_sql_multi

def test_load_sample_sales():
    df = load_dataframe("data/sample_sales.csv")
    assert len(df) > 0 and "revenue" in df.columns

def test_valid_select_query():
    df = load_dataframe("data/sample_sales.csv")
    res = run_sql(df, "SELECT region, SUM(revenue) AS total_rev FROM data GROUP BY region")
    assert len(res) == 4

def test_blocks_destructive_sql():
    df = load_dataframe("data/sample_sales.csv")
    with pytest.raises(ValueError):
        run_sql(df, "DROP TABLE data")