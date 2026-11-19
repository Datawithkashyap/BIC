import html
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor
)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    precision_score, r2_score, recall_score, roc_auc_score,
    root_mean_squared_error, silhouette_score
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def handle_missing_and_outliers(
    df: pd.DataFrame,
    missing_strategy: str = "Auto (Skew-Aware)",
    outlier_strategy: str = "IQR Winsorize (Cap 1.5xIQR)",
) -> tuple[pd.DataFrame, list[str]]:
    log = []
    out = df.copy()
    out = out.dropna(how="all").dropna(axis=1, how="all")

    for col in out.select_dtypes(include="object").columns:
        out[col] = out[col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan, "None": np.nan})
        cleaned = out[col].dropna().str.replace(r"[$,%]", "", regex=True)
        converted = pd.to_numeric(cleaned, errors="coerce")
        if len(cleaned) > 0 and converted.notna().mean() > 0.85:
            out[col] = pd.to_numeric(out[col].str.replace(r"[$,%]", "", regex=True), errors="coerce")
            log.append(f"Coerced '{col}' from text to numeric.")

    dup_cnt = int(out.duplicated().sum())
    if dup_cnt > 0:
        out = out.drop_duplicates()
        log.append(f"Dropped {dup_cnt} duplicate rows.")

    if missing_strategy == "Auto (Skew-Aware)":
        high_null = [c for c in out.columns if out[c].isna().mean() > 0.50]
        if high_null:
            out = out.drop(columns=high_null)
            log.append(f"Dropped high-null (>50%) columns: {', '.join(high_null)}.")

    missing_before = int(out.isna().sum().sum())
    if missing_before > 0:
        num_cols = out.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in out.columns if c not in num_cols and not pd.api.types.is_datetime64_any_dtype(out[c])]
        if missing_strategy == "Drop Missing Rows":
            out = out.dropna()
            log.append(f"Dropped rows with missing values ({missing_before} missing cells removed).")
        else:
            if missing_strategy == "KNN Imputer" and num_cols:
                out[num_cols] = KNNImputer(n_neighbors=5).fit_transform(out[num_cols])
                log.append("Imputed numeric nulls with KNNImputer (k=5).")
            else:
                for col in num_cols:
                    if out[col].isna().any():
                        if missing_strategy == "Mean / Mode":
                            fill_val, method = out[col].mean(), "mean"
                        elif missing_strategy == "Median / Mode":
                            fill_val, method = out[col].median(), "median"
                        else:
                            skew = abs(float(out[col].skew())) if out[col].dropna().shape[0] > 3 else 0.0
                            fill_val = out[col].median() if skew > 1.0 else out[col].mean()
                            method = "median (skewed)" if skew > 1.0 else "mean (symmetric)"
                        out[col] = out[col].fillna(fill_val)
                        log.append(f"Filled '{col}' nulls using {method}.")
            for col in cat_cols:
                if out[col].isna().any():
                    mode_val = out[col].mode()
                    out[col] = out[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")
                    log.append(f"Filled '{col}' nulls using mode.")

    num_cols = [c for c in out.select_dtypes(include="number").columns if out[c].nunique() > 10]
    if outlier_strategy == "IQR Winsorize (Cap 1.5xIQR)" and num_cols:
        capped = 0
        for col in num_cols:
            q1, q3 = out[col].quantile(0.25), out[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                cnt = int(((out[col] < lo) | (out[col] > hi)).sum())
                if cnt > 0:
                    out[col] = out[col].clip(lo, hi)
                    capped += cnt
        if capped:
            log.append(f"Winsorized {capped} outlier values using 1.5×IQR bounds.")
    elif outlier_strategy == "Z-Score Cap (|Z| > 3)" and num_cols:
        capped = 0
        for col in num_cols:
            mu, sd = out[col].mean(), out[col].std()
            if sd > 0:
                lo, hi = mu - 3 * sd, mu + 3 * sd
                cnt = int(((out[col] < lo) | (out[col] > hi)).sum())
                if cnt > 0:
                    out[col] = out[col].clip(lo, hi)
                    capped += cnt
        if capped:
            log.append(f"Capped {capped} outlier values at ±3σ (Z-Score).")
    elif outlier_strategy == "Drop Outlier Rows (IQR)" and num_cols:
        mask = pd.Series(True, index=out.index)
        for col in num_cols:
            q1, q3 = out[col].quantile(0.25), out[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                mask &= (out[col] >= q1 - 1.5 * iqr) & (out[col] <= q3 + 1.5 * iqr)
        dropped = int((~mask).sum())
        if dropped:
            out = out[mask]
            log.append(f"Removed {dropped} rows outside 1.5×IQR bounds.")

    if not log:
        log.append("Dataset is clean—no changes needed.")
    return out, log

def profile_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        missing = int(s.isna().sum())
        outlier_cnt, skew_val = 0, None
        if pd.api.types.is_numeric_dtype(s) and s.dropna().shape[0] > 3:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            outlier_cnt = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            skew_val = round(float(s.skew()), 2)
        rows.append({
            "column": col, "dtype": str(s.dtype), "missing": missing,
            "missing_%": round(100 * missing / max(len(df), 1), 1),
            "unique": int(s.nunique(dropna=True)),
            "iqr_outliers": outlier_cnt, "skewness": skew_val,
        })
    return pd.DataFrame(rows)

def run_hypothesis_test(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    sub = df[[col_a, col_b]].dropna()
    a_num = pd.api.types.is_numeric_dtype(sub[col_a])
    b_num = pd.api.types.is_numeric_dtype(sub[col_b])
    if a_num and b_num:
        stat, p = stats.pearsonr(sub[col_a], sub[col_b])
        test_name = "Pearson Correlation"
    elif (not a_num and b_num) or (a_num and not b_num):
        cat_c, num_c = (col_a, col_b) if not a_num else (col_b, col_a)
        groups = [g[num_c].values for _, g in sub.groupby(cat_c) if len(g) > 1]
        if len(groups) == 2:
            stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            test_name = "Welch's t-test"
        else:
            stat, p = stats.f_oneway(*groups)
            test_name = "One-Way ANOVA"
    else:
        ct = pd.crosstab(sub[col_a], sub[col_b])
        stat, p, _, _ = stats.chi2_contingency(ct)
        test_name = "Chi-Square Test of Independence"
    return {"test": test_name, "statistic": round(float(stat), 4), "p_value": round(float(p), 6), "significant": bool(p < 0.05)}

def run_supervised_ml(df: pd.DataFrame, target_col: str, model_choice: str) -> dict:
    data = df.dropna(subset=[target_col]).copy()
    X = data.drop(columns=[target_col])
    y = data[target_col]
    drop_cols = [c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c]) or (X[c].dtype == "object" and X[c].nunique() > 40)]
    X = X.drop(columns=drop_cols)

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])

    is_clf = (not pd.api.types.is_numeric_dtype(y)) or (y.nunique() <= 10)
    if is_clf:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
        pipe = Pipeline([("pre", pre), ("model", models[model_choice])])
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        cv_scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring="accuracy")
        metrics = {
            "Accuracy": round(float(accuracy_score(y_te, preds)), 4),
            "Precision": round(float(precision_score(y_te, preds, average="weighted", zero_division=0)), 4),
            "Recall": round(float(recall_score(y_te, preds, average="weighted", zero_division=0)), 4),
            "F1 Score": round(float(f1_score(y_te, preds, average="weighted", zero_division=0)), 4),
            "5-Fold CV": round(float(cv_scores.mean()), 4),
        }
        if y.nunique() == 2 and hasattr(pipe, "predict_proba"):
            metrics["ROC-AUC"] = round(float(roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])), 4)
        labels = sorted(y.unique().tolist(), key=str)
        extra = {"confusion_matrix": pd.DataFrame(confusion_matrix(y_te, preds, labels=labels), index=labels, columns=labels)}
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }
        pipe = Pipeline([("pre", pre), ("model", models[model_choice])])
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        cv_scores = cross_val_score(pipe, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring="r2")
        metrics = {
            "R2 Score": round(float(r2_score(y_te, preds)), 4),
            "MAE": round(float(mean_absolute_error(y_te, preds)), 4),
            "RMSE": round(float(root_mean_squared_error(y_te, preds)), 4),
            "5-Fold CV R2": round(float(cv_scores.mean()), 4),
        }
        extra = {"actual_vs_pred": pd.DataFrame({"Actual": y_te.values, "Predicted": preds})}

    feat_names = pipe.named_steps["pre"].get_feature_names_out()
    model_obj = pipe.named_steps["model"]
    imp = model_obj.feature_importances_ if hasattr(model_obj, "feature_importances_") else (np.abs(model_obj.coef_).ravel()[:len(feat_names)] if hasattr(model_obj, "coef_") else np.zeros(len(feat_names)))
    fi = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False).head(12)
    return {
        "task": "Classification" if is_clf else "Regression",
        "metrics": metrics,
        "feature_importance": fi,
        "pipeline": pipe,
        "feature_columns": X.columns.tolist(),
        **extra,
    }

def run_kmeans_pca(df: pd.DataFrame, k: int = 3) -> dict:
    num_df = df.select_dtypes(include="number").dropna()
    scaled = StandardScaler().fit_transform(num_df)
    clusters = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(scaled)
    sil = round(float(silhouette_score(scaled, clusters)), 4) if k > 1 and len(num_df) > k else 0.0
    pca = PCA(n_components=2, random_state=42).fit_transform(scaled)
    return {"silhouette": sil, "plot_df": pd.DataFrame({"PC1": pca[:, 0], "PC2": pca[:, 1], "Cluster": clusters.astype(str)})}

def build_executive_report_html(table_name: str, df: pd.DataFrame, audit_log: list[str], chat_history: list[dict]) -> str:
    profile_html = profile_dataset(df).to_html(index=False, border=0)
    audit_items = "".join(f"<li>{html.escape(m)}</li>" for m in audit_log)
    insights_html = "".join(
        f"<div class='card'><h4>Q: {html.escape(h['question'])}</h4>"
        f"<pre>{html.escape(h.get('sql', ''))}</pre>"
        f"<p>{html.escape(h.get('insight', ''))}</p></div>"
        for h in chat_history
    ) or "<p>No SQL Q&amp;A queries run in this session yet.</p>"

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>BIC Executive Brief — {html.escape(table_name)}</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 40px auto; color: #0F172A; line-height: 1.5; }}
h1, h2 {{ border-bottom: 2px solid #0F766E; padding-bottom: 6px; }}
table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13px; }}
th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #CBD5E1; }}
.card {{ background: #F8FAFC; border-left: 4px solid #0F766E; padding: 12px 16px; margin: 12px 0; }}
pre {{ background: #0F172A; color: #E2E8F0; padding: 10px; overflow-x: auto; font-size: 12px; }}
</style></head><body>
<h1>BIC Executive Brief — {html.escape(table_name)}</h1>
<p><strong>Rows:</strong> {df.shape[0]} &nbsp;|&nbsp; <strong>Columns:</strong> {df.shape[1]}</p>
<h2>Data Preparation &amp; Audit Log</h2><ul>{audit_items}</ul>
<h2>Column Quality Profile</h2>{profile_html}
<h2>SQL Copilot Findings</h2>{insights_html}
</body></html>"""