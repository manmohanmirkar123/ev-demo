"""Live fleet-health dashboard (reads DynamoDB). Run: streamlit run dashboard/app.py"""
import json
import pathlib

import boto3
import pandas as pd
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent.parent
cfg = {k: v["value"] for k, v in json.loads((ROOT / "config.json").read_text()).items()}
session = boto3.Session(region_name=cfg["region"])
ddb = session.resource("dynamodb")
s3 = session.client("s3")
health_table = ddb.Table(cfg["health_table"])
wo_table = ddb.Table(cfg["work_orders_table"])

STATUS_STYLE = {
    "HEALTHY": "background-color:#d9f2e3;color:#14532d",
    "AT_RISK": "background-color:#fde9c8;color:#7c4a03",
    "CRITICAL": "background-color:#f9d0d0;color:#7f1d1d",
}


def model_info():
    """Which SageMaker training job produced the model the scorer is using, if any."""
    try:
        body = s3.get_object(Bucket=cfg["lake_bucket"], Key="model/model-info.json")["Body"].read()
        return json.loads(body)
    except s3.exceptions.NoSuchKey:
        return None


def scan(table):
    items, kwargs = [], {}
    while True:
        page = table.scan(**kwargs)
        items += page["Items"]
        if "LastEvaluatedKey" not in page:
            return items
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]


st.set_page_config(page_title="EV Predictive Maintenance", layout="wide")
st.title("EV fleet — predictive maintenance")
st.caption("Simulated fleet → AWS IoT Core → Lambda scorer → DynamoDB · risk events → Step Functions work orders")


@st.fragment(run_every=3)
def live():
    items = scan(health_table)
    if not items:
        st.info("No telemetry yet — start the simulator with `make simulate`.")
        return

    df = pd.DataFrame(items)
    for col in ("rul_source", "rul_trend_days", "rul_model_days"):
        if col not in df:
            df[col] = None
    for col in ("value", "rul_days", "rul_trend_days", "rul_model_days", "z", "threshold", "sim_day"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    orders = pd.DataFrame(scan(wo_table))

    info = model_info()
    if info:
        mae = info.get("metrics", {}).get("validation:mae")
        where = ("XGBoost model trained **locally** (no SageMaker)" if info.get("trained_with") == "local"
                 else f"XGBoost model trained on **SageMaker** · job `{info['training_job']}`"
                 + (f" · account {info['account']}" if info.get("account") else ""))
        st.success(f"Days-to-limit: {where}" + (f" · validation MAE {mae:.1f} days" if mae is not None else ""))
    else:
        st.warning("Days-to-limit: **linear trend**, no ML model published "
                   "(`make train-local`, or `make train` with SageMaker)")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Simulated day", int(df["sim_day"].max()))
    c2.metric("Vehicles", df["vehicle_id"].nunique())
    c3.metric("Parts at risk", int((df["status"] == "AT_RISK").sum()))
    c4.metric("Parts critical", int((df["status"] == "CRITICAL").sum()))
    c5.metric("Work orders", len(orders))

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Fleet health matrix")
        grid = df.pivot(index="vehicle_id", columns="label", values="status").fillna("")
        st.dataframe(grid.style.map(lambda s: STATUS_STYLE.get(s, "")), height=740)

    with right:
        st.subheader("Needs attention")
        risky = (df[df["status"] != "HEALTHY"]
                 .assign(_rank=lambda d: d["status"].map({"CRITICAL": 0, "AT_RISK": 1}))
                 .sort_values(["_rank", "rul_days"]))
        if risky.empty:
            st.success("All components healthy.")
        else:
            show = risky[["vehicle_id", "label", "status", "value", "unit", "rul_days", "rul_model_days",
                          "rul_trend_days", "z"]].rename(
                columns={"label": "component", "rul_days": "days to limit", "rul_model_days": "model",
                         "rul_trend_days": "trend", "z": "anomaly z"})
            st.dataframe(show.style.map(lambda s: STATUS_STYLE.get(s, ""), subset=["status"]),
                         hide_index=True)

        st.subheader("Service work orders")
        if orders.empty:
            st.caption("None yet.")
        else:
            cols = ["created_at", "priority", "vehicle_id", "component", "reading", "rul_days", "book_within"]
            st.dataframe(orders.sort_values("created_at", ascending=False)[cols], hide_index=True)

    st.subheader("Component trend")
    vehicles = sorted(df["vehicle_id"].unique())
    default = vehicles.index(risky["vehicle_id"].iloc[0]) if not risky.empty else 0
    vid = st.selectbox("Vehicle", vehicles, index=default)
    cols = st.columns(5)
    for col, (_, row) in zip(cols, df[df["vehicle_id"] == vid].sort_values("label").iterrows()):
        hist = pd.DataFrame([(int(d), float(v)) for d, v in row["history"]], columns=["day", row["unit"]])
        hist["limit"] = float(row["threshold"])
        with col:
            st.markdown(f"**{row['label']}** · {row['status']}")
            st.line_chart(hist.set_index("day"), height=180)


live()
