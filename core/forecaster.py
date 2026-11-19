import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

def forecast_metric(df: pd.DataFrame, date_col: str, value_col: str, periods: int = 6, freq: str = "MS") -> pd.DataFrame:
    ts = df[[date_col, value_col]].dropna().copy()
    ts[date_col] = pd.to_datetime(ts[date_col])
    ts = ts.groupby(date_col, as_index=False)[value_col].sum().sort_values(date_col)
    ts["unique_id"] = "series_1"
    ts = ts.rename(columns={date_col: "ds", value_col: "y"})

    sf = StatsForecast(models=[AutoARIMA()], freq=freq)
    sf.fit(ts)
    preds = sf.predict(h=periods, level=[90])
    return preds.reset_index(drop=True)