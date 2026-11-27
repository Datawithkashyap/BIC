import io
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from core.sql_engine import (
    clean_table_name, get_multi_schema, load_dataframe, run_sql_multi
)
from core.llm_copilot import generate_sql, repair_sql, summarize_insight, warm_ollama
from core.forecaster import forecast_metric
from core.telemetry import get_telemetry_summary
from core.analyst_workbench import (
    build_executive_report_html, handle_missing_and_outliers, profile_dataset,
    run_hypothesis_test, run_kmeans_pca, run_supervised_ml
)

st.set_page_config(page_title="BIC — Business Intelligence Copilot", layout="wide")
st.title("BIC — AI-Powered Business Intelligence Copilot")

@st.cache_resource
def init_ollama_once():
    return warm_ollama()

init_ollama_once()

@st.cache_data
def cached_load(file_bytes: bytes | None, file_name: str) -> pd.DataFrame:
    if file_bytes is None:
        return load_dataframe("data/sample_sales.csv")
    buf = io.BytesIO(file_bytes)
    buf.name = file_name
    return load_dataframe(buf)

uploaded_files = st.file_uploader("Upload 1 or more CSV/Excel files", type=["csv", "xlsx"], accept_multiple_files=True)

with st.expander("Upload & Clean Dataset (CSV / Excel)", expanded=False):
    st.subheader("Data Cleaning & Outlier Controls")
    missing_opt = st.selectbox("Missing Values Procedure", [
    "Auto (Skew-Aware)", "Median / Mode", "Mean / Mode", "KNN Imputer", "Drop Missing Rows"
])
    outlier_opt = st.selectbox("Outlier Management Procedure", [
    "IQR Winsorize (Cap 1.5xIQR)", "Z-Score Cap (|Z| > 3)", "Drop Outlier Rows (IQR)", "None"
])

raw_tables = {}
if uploaded_files:
    for f in uploaded_files:
        raw_tables[clean_table_name(f.name)] = cached_load(f.getvalue(), f.name)
else:
    raw_tables["sample_sales"] = cached_load(None, "sample_sales.csv")

tables = {}
audit_logs = {}
for tname, rdf in raw_tables.items():
    tables[tname], audit_logs[tname] = handle_missing_and_outliers(rdf, missing_opt, outlier_opt)

active_table = st.sidebar.selectbox("Active Table for EDA / Visuals / ML", list(tables.keys()))
df = tables[active_table]
audit_log = audit_logs[active_table]
schema = get_multi_schema(tables)

st.sidebar.metric("Active Rows Ã— Columns", f"{df.shape[0]} Ã— {df.shape[1]}")
with st.sidebar.expander("Cleaning & Outlier Audit Log"):
    for msg in audit_log:
        st.write(f"â€¢ {msg}")
st.sidebar.code(schema)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tab_qa, tab_eda, tab_viz, tab_ml, tab_guide = st.tabs([
    "Ask BIC (Chat + SQL)",
    "Auto-EDA & Stats",
    "Visualization Studio",
    "ML Lab & Forecasting",
    "Training Guide",
])

with tab_qa:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    prompts = [f"Show row count and summary of {active_table}"]
    if cat_cols and num_cols:
        prompts.append(f"Show average {num_cols[0]} by {cat_cols[0]}")
        prompts.append(f"Top 5 {cat_cols[0]} by total {num_cols[0]}")

    p_cols = st.columns(len(prompts))
    clicked_prompt = None
    for idx, p_text in enumerate(prompts):
        if p_cols[idx].button(p_text, key=f"chip_{idx}"):
            clicked_prompt = p_text

    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(msg["question"])
        with st.chat_message("assistant"):
            edited_sql = st.text_area("SQL Query (editable)", value=msg["sql"], key=f"sql_edit_{idx}", height=90)
            if st.button("Run Edited SQL", key=f"run_edit_{idx}"):
                try:
                    msg["df"] = run_sql_multi(tables, edited_sql)
                    msg["sql"] = edited_sql
                except Exception as err:
                    st.error(f"SQL Error: {err}")
            st.dataframe(msg["df"], use_container_width=True)
            st.info(msg["insight"])

    user_q = st.chat_input("Ask a question or follow-up (e.g. 'Now filter only top 3')...") or clicked_prompt
    if user_q:
        with st.spinner("Generating SQL and insights..."):
            try:
                sql = generate_sql(user_q, schema, st.session_state.chat_history)
                try:
                    result = run_sql_multi(tables, sql)
                except Exception as sql_err:
                    sql = repair_sql(user_q, schema, sql, str(sql_err))
                    result = run_sql_multi(tables, sql)
                insight = summarize_insight(user_q, result.head(15).to_string(index=False))
                st.session_state.chat_history.append({"question": user_q, "sql": sql, "df": result, "insight": insight})
                st.rerun()
            except Exception as exc:
                st.error(f"Could not complete query: {exc}")

    brief_html = build_executive_report_html(active_table, df, audit_log, st.session_state.chat_history)
    st.download_button("Export Executive Brief (HTML)", brief_html.encode("utf-8"), f"{active_table}_executive_brief.html", "text/html")

with tab_eda:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", len(df))
    c2.metric("Missing Cells", int(df.isna().sum().sum()))
    c3.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.subheader("Column Profile, IQR Outliers & Skewness")
    st.dataframe(profile_dataset(df), use_container_width=True)

    st.subheader("Statistical Hypothesis Testing")
    col_a = st.selectbox("Variable A", df.columns.tolist(), key="hyp_a")
    col_b = st.selectbox("Variable B", df.columns.tolist(), index=min(1, len(df.columns) - 1), key="hyp_b")
    if st.button("Run Significance Test"):
        res = run_hypothesis_test(df, col_a, col_b)
        st.write(f"**{res['test']}** — Statistic: `{res['statistic']}`, p-value: `{res['p_value']}`")
        st.success("Statistically significant (p < 0.05)" if res["significant"] else "Not statistically significant (p >= 0.05)")

    st.download_button("Download Cleaned CSV", df.to_csv(index=False).encode("utf-8"), f"{active_table}_cleaned.csv", "text/csv")

with tab_viz:
    viz_tool = st.selectbox("Visualization Library", ["Plotly (Interactive)", "Seaborn / Matplotlib (Statistical)"])
    chart_type = st.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Violin Plot", "Pie / Donut Chart", "Correlation Heatmap"])
    all_cols = df.columns.tolist()
    v_num_cols = df.select_dtypes(include="number").columns.tolist()

    if viz_tool == "Plotly (Interactive)":
        if chart_type == "Correlation Heatmap" and len(v_num_cols) >= 2:
            st.plotly_chart(px.imshow(df[v_num_cols].corr(numeric_only=True), text_auto=True, title="Correlation Matrix"), use_container_width=True)
        elif chart_type == "Histogram":
            c = st.selectbox("Column", v_num_cols or all_cols)
            st.plotly_chart(px.histogram(df, x=c, marginal="box", title=f"Distribution of {c}"), use_container_width=True)
        elif chart_type in ("Box Plot", "Violin Plot"):
            y_c = st.selectbox("Numeric Metric", v_num_cols or all_cols)
            x_c = st.selectbox("Group By", ["None"] + all_cols)
            fn = px.box if chart_type == "Box Plot" else px.violin
            st.plotly_chart(fn(df, y=y_c, x=None if x_c == "None" else x_c, title=f"{chart_type}: {y_c}"), use_container_width=True)
        elif chart_type == "Pie / Donut Chart":
            n_c = st.selectbox("Category", all_cols)
            v_c = st.selectbox("Values", v_num_cols or all_cols)
            st.plotly_chart(px.pie(df, names=n_c, values=v_c, hole=0.4, title=f"{v_c} Share by {n_c}"), use_container_width=True)
        else:
            x_c = st.selectbox("X-Axis", all_cols)
            y_c = st.selectbox("Y-Axis", v_num_cols or all_cols)
            c_c = st.selectbox("Color (optional)", ["None"] + all_cols)
            c_arg = None if c_c == "None" else c_c
            if chart_type == "Bar Chart":
                st.plotly_chart(px.bar(df, x=x_c, y=y_c, color=c_arg, title=f"{y_c} by {x_c}"), use_container_width=True)
            elif chart_type == "Line Chart":
                st.plotly_chart(px.line(df.sort_values(x_c), x=x_c, y=y_c, color=c_arg, title=f"{y_c} over {x_c}"), use_container_width=True)
            else:
                st.plotly_chart(px.scatter(df, x=x_c, y=y_c, color=c_arg, title=f"{y_c} vs {x_c}"), use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if chart_type == "Correlation Heatmap" and len(v_num_cols) >= 2:
            sns.heatmap(df[v_num_cols].corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        elif chart_type == "Histogram":
            c = st.selectbox("Column", v_num_cols or all_cols, key="sb_h")
            sns.histplot(df[c], kde=True, ax=ax)
        elif chart_type in ("Box Plot", "Violin Plot"):
            y_c = st.selectbox("Numeric Metric", v_num_cols or all_cols, key="sb_y")
            x_c = st.selectbox("Group By", ["None"] + all_cols, key="sb_x")
            fn = sns.boxplot if chart_type == "Box Plot" else sns.violinplot
            fn(data=df, y=y_c, x=None if x_c == "None" else x_c, ax=ax)
        else:
            x_c = st.selectbox("X-Axis", all_cols, key="sb_ax")
            y_c = st.selectbox("Y-Axis", v_num_cols or all_cols, key="sb_ay")
            if chart_type == "Bar Chart":
                sns.barplot(data=df, x=x_c, y=y_c, ax=ax)
            elif chart_type == "Line Chart":
                sns.lineplot(data=df.sort_values(x_c), x=x_c, y=y_c, ax=ax)
            else:
                sns.scatterplot(data=df, x=x_c, y=y_c, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)

with tab_ml:
    ml_mode = st.selectbox("ML Workflow", ["Supervised ML + What-If Simulator", "Unsupervised Clustering (K-Means + PCA)", "Time-Series Forecasting (AutoARIMA)"])
    if ml_mode == "Supervised ML + What-If Simulator":
        target = st.selectbox("Target Column", df.columns.tolist())
        is_clf = (not pd.api.types.is_numeric_dtype(df[target])) or (df[target].nunique() <= 10)
        algo = st.selectbox("Model", ["Logistic Regression", "Random Forest", "Gradient Boosting"] if is_clf else ["Linear Regression", "Random Forest", "Gradient Boosting"])
        if st.button("Train & Validate (5-Fold CV)"):
            st.session_state.trained_ml = run_supervised_ml(df, target, algo)

        if "trained_ml" in st.session_state:
            res = st.session_state.trained_ml
            st.success(f"{res['task']} Model Ready")
            cols = st.columns(len(res["metrics"]))
            for idx, (k, v) in enumerate(res["metrics"].items()):
                cols[idx].metric(k, v)
            st.plotly_chart(px.bar(res["feature_importance"], x="importance", y="feature", orientation="h", title="Top Feature Importances"), use_container_width=True)
            if "confusion_matrix" in res:
                st.plotly_chart(px.imshow(res["confusion_matrix"], text_auto=True, title="Confusion Matrix"), use_container_width=True)

            st.subheader("What-If Prediction Simulator")
            sim_input = {}
            s_cols = st.columns(3)
            for idx, col_name in enumerate(res["feature_columns"]):
                with s_cols[idx % 3]:
                    if pd.api.types.is_numeric_dtype(df[col_name]):
                        sim_input[col_name] = st.number_input(col_name, value=float(df[col_name].median()))
                    else:
                        opts = df[col_name].dropna().unique().tolist()[:30]
                        sim_input[col_name] = st.selectbox(col_name, opts, key=f"sim_{col_name}")
            if st.button("Predict Scenario"):
                pred = res["pipeline"].predict(pd.DataFrame([sim_input]))[0]
                st.metric("Predicted Outcome", str(pred))

    elif ml_mode == "Unsupervised Clustering (K-Means + PCA)":
        k = st.slider("Clusters (k)", 2, 8, 3)
        if st.button("Run K-Means + PCA"):
            res = run_kmeans_pca(df, k=k)
            st.metric("Silhouette Score", res["silhouette"])
            st.plotly_chart(px.scatter(res["plot_df"], x="PC1", y="PC2", color="Cluster", title="2D PCA Cluster Projection"), use_container_width=True)
    else:
        date_cols = [c for c in df.columns if "date" in c or "time" in c]
        f_num_cols = df.select_dtypes(include="number").columns.tolist()
        if date_cols and f_num_cols:
            d_col = st.selectbox("Date column", date_cols)
            v_col = st.selectbox("Metric", f_num_cols)
            freq = st.selectbox("Frequency", ["MS", "W", "D"])
            horizon = st.slider("Periods ahead", 3, 12, 6)
            if st.button("Generate Forecast"):
                preds = forecast_metric(df, d_col, v_col, periods=horizon, freq=freq)
                st.dataframe(preds, use_container_width=True)
                st.plotly_chart(px.line(preds, x="ds", y="AutoARIMA", title=f"{v_col} Forecast"), use_container_width=True)
        else:
            st.warning("Upload a dataset with a date column and a numeric column to forecast.")

with tab_guide:
    st.subheader("Welcome to BIC (Business Intelligence Copilot)")
    st.write("Follow these steps to go from raw spreadsheets to SQL, ML predictions, and executive briefs:")

    g_num = df.select_dtypes(include="number").columns.tolist()
    g_cat = [c for c in df.columns if c not in g_num]

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("""
**Complete BIC Workflow**
1. **Upload & Clean (`Upload & Clean Dataset`):** Drop one or multiple `.csv`/`.xlsx` files for DuckDB `JOIN`s, then pick **Auto (Skew-Aware)** / **KNN Imputer** and **IQR Winsorize** / **Z-Score** outlier handling.
2. **Auto-EDA & Stats:** Inspect missing %, skewness, IQR outliers, and run **Welch's t-test, ANOVA, Chi-Square, or Pearson correlation**.
3. **Ask BIC (Chat + SQL):** Ask follow-up questions, edit SQL inline, rely on `repair_sql()` self-healing, or click **Export Executive Brief (HTML)**.
4. **Visualization & ML Lab:** Build 8 chart types in **Plotly** or **Seaborn/Matplotlib**, train 5-Fold CV models with the **What-If Simulator**, run **K-Means + PCA**, or forecast with **AutoARIMA**.
5. **REST API (`api.py`):** Serve `/health`, `/query`, and `/profile` via FastAPI (`uvicorn api:app --port 8000`).
""")
    with c_right:
        st.markdown(f"**Live Sample Prompts for `{active_table}`**")
        dyn_prompts = [f"Show total row count and summary of {active_table}"]
        if g_cat and g_num:
            dyn_prompts.append(f"Show total {g_num[0]} and average {g_num[0]} by {g_cat[0]}")
            dyn_prompts.append(f"Top 5 {g_cat[0]} by {g_num[0]} descending")
        if len(g_num) >= 2:
            dyn_prompts.append(f"Compare {g_num[0]} and {g_num[1]} across {active_table}")
        for dp in dyn_prompts:
            st.markdown(f'- *"{dp}"*')

        st.divider()
        st.markdown("**Live System Telemetry (`logs/query_metrics.csv`)**")
        t_stats = get_telemetry_summary()
        t1, t2, t3 = st.columns(3)
        t1.metric("Logged Queries", t_stats["queries"])
        t2.metric("Avg Latency", f"{t_stats['avg_latency_s']}s")
        t3.metric("SQL Repair Rate", f"{t_stats['repair_rate_pct']}%")
