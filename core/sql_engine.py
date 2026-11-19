import os
import re
import duckdb
import pandas as pd

def clean_table_name(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", base).strip("_").lower()
    return cleaned or "data"

def load_dataframe(file_obj_or_path) -> pd.DataFrame:
    name = getattr(file_obj_or_path, "name", str(file_obj_or_path)).lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(file_obj_or_path)
    else:
        df = pd.read_csv(file_obj_or_path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    for col in df.columns:
        if "date" in col or "time" in col:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass
    return df

def get_schema_summary(df: pd.DataFrame, table_name: str = "data") -> str:
    cols = [f"{col} ({dtype})" for col, dtype in df.dtypes.items()]
    return f"Table: {table_name}\nColumns: " + ", ".join(cols)

def get_multi_schema(tables: dict[str, pd.DataFrame]) -> str:
    return "\n\n".join(get_schema_summary(tdf, tname) for tname, tdf in tables.items())

def run_sql(df: pd.DataFrame, sql: str) -> pd.DataFrame:
    return run_sql_multi({"data": df}, sql)

def run_sql_multi(tables: dict[str, pd.DataFrame], sql: str) -> pd.DataFrame:
    clean = sql.strip().rstrip(";")
    if not clean.lower().startswith(("select", "with")):
        raise ValueError("Only read-only SELECT/WITH queries are allowed.")
    con = duckdb.connect(database=":memory:")
    for idx, (tname, tdf) in enumerate(tables.items()):
        con.register(tname, tdf)
        if idx == 0 and "data" not in tables:
            con.register("data", tdf)
    return con.execute(clean).fetchdf()