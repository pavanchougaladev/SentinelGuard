"""SentinelGuard · real-time fraud intelligence dashboard.

A professional analyst console for the SentinelGuard scoring stack:
overview KPIs, a live decision feed, a transaction scoring console with SHAP
reasons, held-out model evaluation, and PSI drift monitoring.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard_data import (  # noqa: E402
    API_URL,
    BATCH_MAX_ROWS,
    DECISION_APPROVE,
    DECISION_BLOCK,
    MODEL_SCHEMAS,
    PSI_ALERT,
    PSI_WARN,
    api_health,
    artifact_age,
    batch_template,
    clear_prediction_cache,
    dataset_mode,
    detect_upload_schema,
    drift_rows,
    fetch_drift,
    load_eval_analytics,
    load_metrics,
    load_policy,
    load_predictions,
    load_reference,
    load_traffic_sample,
    policy_rows,
    prepare_upload,
    row_to_payload,
    score_band_counts,
    score_batch,
    score_transaction,
    shap_image_path,
    upload_row_to_payload,
)
from src.features import COUNTRIES, MCC, TYPES  # noqa: E402

st.set_page_config(
    page_title="SentinelGuard · Fraud intelligence",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATUS_COLORS = {"ok": "#34D399", "WARN": "#FBBF24", "ALERT": "#F87171"}
STATUS_LABELS = {"ok": "Stable", "WARN": "Warning", "ALERT": "Alert"}
BRAND_RED = "#F87171"
BRAND_GREEN = "#34D399"
BRAND_BLUE = "#60A5FA"

KAGGLE_SLIDERS = ["V1", "V2", "V3", "V4"]
KAGGLE_ADVANCED = ["V10", "V12", "V14", "V17", "V20", "V26", "V27"]

KAGGLE_PRESETS: dict[str, dict[str, float]] = {
    "Typical purchase": {
        "amount": 85.0,
        "V1": 0.2,
        "V2": -0.1,
        "V3": 0.4,
        "V4": 0.1,
        **{name: 0.0 for name in KAGGLE_ADVANCED},
    },
    "Card testing": {
        "amount": 1.5,
        "V1": -3.2,
        "V2": 2.6,
        "V3": -4.1,
        "V4": 2.2,
        "V10": -2.0,
        "V12": -4.0,
        "V14": -5.0,
        "V17": -4.5,
        "V20": 1.5,
        "V26": 0.5,
        "V27": 0.5,
    },
    "Account takeover": {
        "amount": 950.0,
        "V1": -5.0,
        "V2": 4.0,
        "V3": -7.0,
        "V4": 3.5,
        "V10": -3.0,
        "V12": -6.0,
        "V14": -9.0,
        "V17": -8.0,
        "V20": 2.0,
        "V26": 1.0,
        "V27": 0.5,
    },
}

SYNTH_PRESETS: dict[str, dict[str, Any]] = {
    "Typical purchase": {
        "amount": 85.0,
        "tx_type": "purchase",
        "merchant_category": "grocery",
        "country": "IN",
        "device_risk": 0.10,
        "ip_risk": 0.05,
        "account_age_days": 900,
        "tx_count_24h": 1,
    },
    "New device, odd hour": {
        "amount": 480.0,
        "tx_type": "online",
        "merchant_category": "electronics",
        "country": "IN",
        "device_risk": 0.55,
        "ip_risk": 0.35,
        "account_age_days": 12,
        "tx_count_24h": 9,
    },
    "Cross-border transfer": {
        "amount": 2400.0,
        "tx_type": "transfer",
        "merchant_category": "crypto",
        "country": "RU",
        "device_risk": 0.85,
        "ip_risk": 0.90,
        "account_age_days": 3,
        "tx_count_24h": 18,
    },
}


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------


def flash(message: str, icon: str = ":material/check_circle:") -> None:
    st.session_state["_flash"] = (message, icon)


def render_flash() -> None:
    item = st.session_state.pop("_flash", None)
    if item:
        st.success(item[0], icon=item[1])


def fmt_money(value: float) -> str:
    return f"${value:,.0f}" if abs(value) >= 1000 else f"${value:,.2f}"


def page_header(title: str, subtitle: str, icon: str) -> None:
    """Consistent page header with a live API badge and a cache refresh."""
    render_flash()
    left, right = st.columns([0.72, 0.28], vertical_alignment="bottom")
    with left:
        st.markdown(f"## {icon} {title}")
        st.caption(subtitle)
    with right:
        online, _ = api_health()
        with st.container(
            horizontal=True,
            horizontal_alignment="right",
            vertical_alignment="center",
        ):
            if online:
                st.badge("API online", icon=":material/check_circle:", color="green")
            else:
                st.badge("API offline", icon=":material/error:", color="red")
            if st.button(
                ":material/refresh: Refresh",
                type="tertiary",
                help="Clear cached data and reload every panel",
            ):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()


SEVERITY_BADGE = {"high": "red", "medium": "orange", "low": "gray"}


def render_reason_codes(codes: list[dict[str, Any]]) -> None:
    """Analyst-facing reason codes with severity-coloured badges."""
    if not codes:
        return
    st.markdown("**Analyst reason codes**")
    for item in codes:
        color = SEVERITY_BADGE.get(str(item.get("severity", "low")), "gray")
        st.markdown(f":{color}-badge[{item['code']}] {item['label']}")
    st.caption(
        "Heuristic summaries of the SHAP drivers above for triage. "
        "They are not causal explanations."
    )


def empty_card(title: str, body: str, icon: str = ":material/inbox:") -> None:
    with st.container(border=True):
        st.markdown(f"#### {icon} {title}")
        st.caption(body)


def decision_badge(decision: str) -> None:
    if decision == DECISION_APPROVE:
        st.badge("Approve", icon=":material/verified_user:", color="green")
    else:
        st.badge("Block / review", icon=":material/gpp_bad:", color="red")


# ---------------------------------------------------------------------------
# Chart builders (one engine: Apache ECharts via st.echarts_chart)
# ---------------------------------------------------------------------------


def risk_band_option(ev: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": {"source": ev["bands"]},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"bottom": 0},
        "grid": {"left": 10, "right": 14, "top": 46, "bottom": 44, "containLabel": True},
        "xAxis": {"type": "category", "axisTick": {"show": False}},
        "yAxis": {
            "type": "value",
            "name": "Share of class (%)",
            "nameGap": 24,
            "axisLabel": {"formatter": "{value}%"},
        },
        "series": [
            {
                "type": "bar",
                "name": "Legitimate",
                "encode": {"x": "band", "y": "legitimate"},
                "barMaxWidth": 26,
                "itemStyle": {"color": BRAND_BLUE, "borderRadius": [4, 4, 0, 0]},
            },
            {
                "type": "bar",
                "name": "Fraud",
                "encode": {"x": "band", "y": "fraud"},
                "barMaxWidth": 26,
                "itemStyle": {"color": BRAND_RED, "borderRadius": [4, 4, 0, 0]},
            },
        ],
    }


def decision_mix_option(approved: int, blocked: int) -> dict[str, Any]:
    return {
        "tooltip": {"trigger": "item", "valueFormatter": "%,d decisions"},
        "legend": {"bottom": 0},
        "series": [
            {
                "type": "pie",
                "radius": ["55%", "78%"],
                "center": ["50%", "44%"],
                "avoidLabelOverlap": True,
                "itemStyle": {"borderRadius": 6, "borderColor": "transparent"},
                "label": {"show": False},
                "emphasis": {"label": {"show": True, "fontWeight": "bold"}},
                "data": [
                    {"value": int(approved), "name": "Approved", "itemStyle": {"color": BRAND_GREEN}},
                    {"value": int(blocked), "name": "Block / review", "itemStyle": {"color": BRAND_RED}},
                ],
            }
        ],
    }


def roc_option(ev: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": {"source": ev["roc"]},
        "tooltip": {"trigger": "axis", "valueFormatter": "percent"},
        "legend": {"bottom": 0},
        "grid": {"left": 12, "right": 18, "top": 18, "bottom": 48, "containLabel": True},
        "xAxis": {
            "type": "value",
            "min": 0,
            "max": 1,
            "name": "False positive rate",
            "nameLocation": "middle",
            "nameGap": 28,
        },
        "yAxis": {
            "type": "value",
            "min": 0,
            "max": 1,
            "name": "True positive rate",
            "nameLocation": "middle",
            "nameGap": 36,
        },
        "series": [
            {
                "type": "line",
                "name": "Model",
                "encode": {"x": "fpr", "y": "tpr"},
                "showSymbol": False,
                "lineStyle": {"width": 2.5},
                "areaStyle": {"opacity": 0.08},
            },
            {
                "type": "line",
                "name": "Random",
                "encode": {"x": "fpr", "y": "baseline"},
                "showSymbol": False,
                "lineStyle": {"width": 1, "type": "dashed"},
                "itemStyle": {"color": "#94A3B8"},
            },
        ],
    }


def pr_option(ev: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": {"source": ev["pr"]},
        "tooltip": {"trigger": "axis", "valueFormatter": "percent"},
        "legend": {"bottom": 0},
        "grid": {"left": 12, "right": 18, "top": 18, "bottom": 48, "containLabel": True},
        "xAxis": {
            "type": "value",
            "min": 0,
            "max": 1,
            "name": "Recall",
            "nameLocation": "middle",
            "nameGap": 28,
        },
        "yAxis": {
            "type": "value",
            "min": 0,
            "max": 1,
            "name": "Precision",
            "nameLocation": "middle",
            "nameGap": 36,
        },
        "series": [
            {
                "type": "line",
                "name": "Model",
                "encode": {"x": "recall", "y": "precision"},
                "showSymbol": False,
                "lineStyle": {"width": 2.5},
                "areaStyle": {"opacity": 0.08},
            },
            {
                "type": "line",
                "name": "Prevalence",
                "encode": {"x": "recall", "y": "prevalence"},
                "showSymbol": False,
                "lineStyle": {"width": 1, "type": "dashed"},
                "itemStyle": {"color": "#94A3B8"},
            },
        ],
    }


def calibration_option(curve: list[dict[str, Any]]) -> dict[str, Any]:
    observed = [
        {
            "predicted": float(row["mean_predicted"]),
            "observed": float(row["fraction_positive"]),
            "count": int(row["count"]),
        }
        for row in curve
        if row.get("mean_predicted") is not None
        and row.get("fraction_positive") is not None
    ]
    return {
        "dataset": {"source": observed},
        "tooltip": {"trigger": "item", "valueFormatter": "percent"},
        "legend": {"bottom": 0},
        "grid": {"left": 12, "right": 18, "top": 18, "bottom": 48, "containLabel": True},
        "xAxis": {
            "type": "value",
            "min": 0,
            "max": 1,
            "name": "Mean predicted score",
            "nameLocation": "middle",
            "nameGap": 28,
        },
        "yAxis": {
            "type": "value",
            "min": 0,
            "max": 1,
            "name": "Observed fraud rate",
            "nameLocation": "middle",
            "nameGap": 36,
        },
        "series": [
            {
                "type": "line",
                "name": "Model",
                "encode": {"x": "predicted", "y": "observed"},
                "symbolSize": 7,
                "lineStyle": {"width": 2.5},
            },
            {
                "type": "line",
                "name": "Perfect calibration",
                "data": [[0, 0], [1, 1]],
                "showSymbol": False,
                "lineStyle": {"width": 1, "type": "dashed"},
                "itemStyle": {"color": "#94A3B8"},
            },
        ],
    }


def confusion_option(cm: dict[str, Any]) -> dict[str, Any]:
    legit = max(cm["tn"] + cm["fp"], 1)
    fraud = max(cm["tp"] + cm["fn"], 1)
    correct = [round(cm["tn"] / legit * 100, 1), round(cm["tp"] / fraud * 100, 1)]
    flagged = [round(cm["fp"] / legit * 100, 1), round(cm["fn"] / fraud * 100, 1)]

    def items(values: list[float]) -> list[dict[str, Any]]:
        # Small segments cannot fit a label; hiding it avoids overprinting the
        # neighbouring segment's percentage.
        return [
            {"value": value, "label": {"show": value >= 5}}
            for value in values
        ]

    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"bottom": 0},
        "grid": {"left": 10, "right": 14, "top": 18, "bottom": 44, "containLabel": True},
        "xAxis": {
            "type": "category",
            "data": ["Legitimate", "Fraud"],
            "axisTick": {"show": False},
        },
        "yAxis": {
            "type": "value",
            "max": 100,
            "axisLabel": {"formatter": "{value}%"},
        },
        "series": [
            {
                "type": "bar",
                "name": "Correctly classified",
                "stack": "total",
                "data": items(correct),
                "itemStyle": {"color": BRAND_GREEN},
                "label": {
                    "show": True,
                    "position": "inside",
                    "formatter": "{c}%",
                    "color": "#FFFFFF",
                },
            },
            {
                "type": "bar",
                "name": "Misclassified",
                "stack": "total",
                "data": items(flagged),
                "itemStyle": {"color": BRAND_RED},
                "label": {
                    "show": True,
                    "position": "inside",
                    "formatter": "{c}%",
                    "color": "#FFFFFF",
                },
            },
        ],
    }


def shap_option(reasons: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [str(item["feature"]) for item in reasons]
    values = [float(item["shap"]) for item in reasons]
    colors = [BRAND_GREEN if value < 0 else BRAND_RED for value in values]
    return {
        "grid": {"left": 6, "right": 42, "top": 8, "bottom": 8, "containLabel": True},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}, "valueFormatter": "%.4f"},
        "xAxis": {"type": "value", "axisLabel": {"formatter": "{value}"}},
        "yAxis": {
            "type": "category",
            "data": labels,
            "inverse": True,
            "axisTick": {"show": False},
        },
        "series": [
            {
                "type": "bar",
                "data": [
                    {"value": value, "itemStyle": {"color": color}}
                    for value, color in zip(values, colors)
                ],
                "barMaxWidth": 22,
                "label": {"show": True, "position": "right", "formatter": "{c}"},
            }
        ],
    }


def psi_option(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = list(reversed(rows))
    return {
        "grid": {"left": 8, "right": 34, "top": 10, "bottom": 10, "containLabel": True},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}, "valueFormatter": "%.4f"},
        "xAxis": {"type": "value", "name": "PSI"},
        "yAxis": {
            "type": "category",
            "data": [str(item["feature"]) for item in ordered],
            "axisTick": {"show": False},
        },
        "series": [
            {
                "type": "bar",
                "data": [
                    {
                        "value": float(item["psi"]),
                        "itemStyle": {"color": STATUS_COLORS.get(item["status"], "#64748B")},
                    }
                    for item in ordered
                ],
                "barMaxWidth": 20,
                "label": {"show": True, "position": "right", "formatter": "{c}"},
                "markLine": {
                    "silent": True,
                    "symbol": "none",
                    "lineStyle": {"type": "dashed", "color": "#F59E0B"},
                    "data": [
                        {"xAxis": PSI_WARN, "label": {"formatter": "warn 0.10"}},
                        {
                            "xAxis": PSI_ALERT,
                            "lineStyle": {"color": BRAND_RED},
                            "label": {"formatter": "alert 0.25"},
                        },
                    ],
                },
            }
        ],
    }


def batch_band_option(bands: pd.DataFrame) -> dict[str, Any]:
    return {
        "dataset": {"source": bands},
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"},
            "valueFormatter": "%,d",
        },
        "legend": {"bottom": 0},
        "grid": {
            "left": 10,
            "right": 14,
            "top": 42,
            "bottom": 44,
            "containLabel": True,
        },
        "xAxis": {"type": "category", "axisTick": {"show": False}},
        "yAxis": {
            "type": "value",
            "name": "Transactions",
            "nameGap": 24,
            "minInterval": 1,
        },
        "series": [
            {
                "type": "bar",
                "name": "Approved",
                "encode": {"x": "band", "y": "approved"},
                "barMaxWidth": 26,
                "itemStyle": {"color": BRAND_GREEN, "borderRadius": [4, 4, 0, 0]},
            },
            {
                "type": "bar",
                "name": "Block / review",
                "encode": {"x": "band", "y": "blocked"},
                "barMaxWidth": 26,
                "itemStyle": {"color": BRAND_RED, "borderRadius": [4, 4, 0, 0]},
            },
        ],
    }


def score_trend_option(df: pd.DataFrame, threshold: float) -> dict[str, Any]:
    return {
        "dataset": {"source": df[["timestamp", "score"]]},
        "tooltip": {"trigger": "axis", "valueFormatter": "percent"},
        "grid": {"left": 10, "right": 16, "top": 16, "bottom": 24, "containLabel": True},
        "xAxis": {"type": "time"},
        "yAxis": {"type": "value", "min": 0, "max": 1},
        "series": [
            {
                "type": "line",
                "name": "Fraud score",
                "encode": {"x": "timestamp", "y": "score"},
                "showSymbol": True,
                "symbolSize": 6,
                "lineStyle": {"width": 2},
                "areaStyle": {"opacity": 0.10},
                "markLine": {
                    "silent": True,
                    "symbol": "none",
                    "lineStyle": {"type": "dashed", "color": BRAND_RED},
                    "data": [{"yAxis": threshold, "label": {"formatter": "threshold"}}],
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# Shared demo traffic
# ---------------------------------------------------------------------------


def simulate_traffic(count: int) -> None:
    """Score real dataset rows through the API to populate the decision log."""
    online, _ = api_health()
    if not online:
        st.error(
            "The scoring API is offline. Start it with "
            "`uvicorn api.main:app --port 8000` and try again.",
            title="Demo traffic unavailable",
            icon=":material/cloud_off:",
        )
        return

    mode = dataset_mode()
    sample = load_traffic_sample(count, seed=int(time.time()))
    if sample.empty:
        st.error(
            "The raw dataset needed for demo traffic was not found in `data/`.",
            title="Demo traffic unavailable",
            icon=":material/folder_off:",
        )
        return

    progress = st.progress(0.0, text="Scoring demo transactions...")
    scored = failed = 0
    for index, (_, row) in enumerate(sample.iterrows(), start=1):
        _, error = score_transaction(row_to_payload(row, mode))
        if error is None:
            scored += 1
        else:
            failed += 1
        progress.progress(index / len(sample), text=f"Scored {index} of {len(sample)}")
    progress.empty()
    clear_prediction_cache()

    # Refresh the drift report immediately so the monitoring page reflects the
    # traffic that was just generated.
    window = int(st.session_state.get("drift_window", 500))
    report, drift_error = fetch_drift(window)
    st.session_state["drift_report"] = report
    st.session_state["drift_error"] = drift_error
    st.session_state["drift_checked_at"] = datetime.now()
    st.session_state["drift_window_used"] = window

    if failed:
        st.warning(f"{scored} transactions scored, {failed} failed.")
    flash(f"Generated {scored} demo decisions and refreshed the drift report.")
    st.rerun()


# ---------------------------------------------------------------------------
# Overview page
# ---------------------------------------------------------------------------


def overview_page() -> None:
    page_header(
        "Fraud operations overview",
        "Decision quality, model health, and live activity across the scoring stack.",
        ":material/space_dashboard:",
    )

    metrics = load_metrics()
    predictions = load_predictions()
    total = len(predictions)
    blocked = (
        int((predictions["decision"] == DECISION_BLOCK).sum())
        if total and "decision" in predictions.columns
        else 0
    )
    block_rate = blocked / total if total else 0.0
    mean_score = (
        float(predictions["score"].mean())
        if total and "score" in predictions.columns
        else 0.0
    )
    recall = float(metrics.get("recall", 0.0))
    precision = float(metrics.get("precision", 0.0))
    pr_auc = float(metrics.get("pr_auc", 0.0))

    with st.container(horizontal=True):
        st.metric(
            "Decisions logged",
            f"{total:,}",
            border=True,
            help="Every transaction scored by the API since the log was created.",
        )
        st.metric(
            "Block / review rate",
            f"{block_rate:.1%}",
            border=True,
            help="Share of logged decisions at or above the operating threshold.",
        )
        st.metric(
            "Mean fraud score",
            f"{mean_score:.1%}",
            border=True,
            help="Average model score across logged decisions.",
        )
        st.metric(
            "Recall",
            f"{recall:.1%}",
            delta=f"{recall - 0.80:+.1%}",
            delta_description="vs 80% target",
            border=True,
            help="Share of real fraud caught on the held-out test split.",
        )
        st.metric(
            "Precision",
            f"{precision:.1%}",
            border=True,
            help="Share of alerts that are real fraud on the held-out test split.",
        )
        st.metric(
            "PR-AUC",
            f"{pr_auc:.3f}",
            border=True,
            help="Area under the precision-recall curve.",
        )

    st.space("small")
    left, right = st.columns([0.62, 0.38], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("**Risk score distribution**")
            st.caption("Share of each class by score band · held-out test split")
            with st.spinner("Scoring the held-out test split..."):
                ev = load_eval_analytics()
            if ev:
                st.echarts_chart(
                    risk_band_option(ev),
                    height=330,
                    alt="Legitimate transactions cluster in low score bands while fraud concentrates above 20%.",
                )
            else:
                st.caption(
                    "Evaluation analytics are unavailable. Check that `models/bundle.joblib` "
                    "and the raw dataset in `data/` are present."
                )

    with right:
        with st.container(border=True):
            st.markdown("**Decision mix**")
            st.caption("Logged decisions at the operating threshold")
            if total:
                st.echarts_chart(
                    decision_mix_option(total - blocked, blocked),
                    height=330,
                    alt=f"{blocked} of {total} logged decisions were blocked or sent for review.",
                )
            else:
                st.caption(
                    "No decisions logged yet. Score a transaction or generate demo "
                    "traffic to populate the feed."
                )
                st.page_link(
                    PAGES["score"],
                    label="Open scoring console",
                    icon=":material/bolt:",
                )

    st.space("small")
    left, right = st.columns([0.62, 0.38], gap="large")

    with left:
        with st.container(border=True):
            header_left, header_right = st.columns([0.7, 0.3], vertical_alignment="center")
            with header_left:
                st.markdown("**Recent decisions**")
            with header_right:
                st.page_link(
                    PAGES["activity"],
                    label="View all",
                    icon=":material/arrow_forward:",
                )
            if predictions.empty:
                st.caption("No decisions logged yet.")
            else:
                recent = predictions.tail(8).iloc[::-1].copy()
                recent["decision_badge"] = recent["decision"].apply(lambda d: [d])
                st.dataframe(
                    recent[["timestamp", "amount", "score", "decision_badge"]],
                    column_config={
                        "timestamp": st.column_config.DatetimeColumn(
                            "Time", format="MMM DD · HH:mm"
                        ),
                        "amount": st.column_config.NumberColumn(
                            "Amount", format="$%,.2f"
                        ),
                        "score": st.column_config.ProgressColumn(
                            "Fraud score",
                            min_value=0.0,
                            max_value=1.0,
                            format="percent",
                        ),
                        "decision_badge": st.column_config.MultiselectColumn(
                            "Decision",
                            options=[DECISION_APPROVE, DECISION_BLOCK],
                            color=["green", "red"],
                        ),
                    },
                    hide_index=True,
                    alt="The eight most recent scored transactions with score and decision.",
                )

    with right:
        with st.container(border=True):
            st.markdown("**Operational status**")
            online, _ = api_health()
            drift = st.session_state.get("drift_report")
            if online:
                st.markdown(":green-badge[Online] Scoring API")
            else:
                st.markdown(":red-badge[Offline] Scoring API")
            if drift and "_retrain_recommended" in drift:
                if drift["_retrain_recommended"]:
                    st.markdown(":red-badge[Alert] Drift monitor")
                else:
                    st.markdown(":green-badge[Stable] Drift monitor")
            else:
                st.markdown(":gray-badge[Not run] Drift monitor")

            st.space("small")
            st.table(
                {
                    "Threshold": f"{float(metrics.get('threshold', 0)):.0%}",
                    "Reference rows": f"{len(load_reference()):,}",
                    "Artifacts updated": artifact_age(),
                },
                border="horizontal",
                width="content",
            )
            st.caption(
                "Refresh clears cached analytics so new artifacts and decisions appear immediately."
            )


# ---------------------------------------------------------------------------
# Score transaction page
# ---------------------------------------------------------------------------


def _init_score_defaults(mode: str) -> None:
    defaults: dict[str, Any]
    if mode == "kaggle":
        defaults = {"amount": 150.0, **{name: 0.0 for name in KAGGLE_SLIDERS}, **{name: 0.0 for name in KAGGLE_ADVANCED}}
    else:
        defaults = dict(SYNTH_PRESETS["Typical purchase"])
    for key, value in defaults.items():
        st.session_state.setdefault(f"score_{key}", value)


def _apply_preset(preset: dict[str, Any]) -> None:
    for key, value in preset.items():
        st.session_state[f"score_{key}"] = value
    st.rerun()


def _render_presets(presets: dict[str, dict[str, Any]]) -> None:
    with st.container(horizontal=True):
        st.caption("Quick profiles")
        for name, preset in presets.items():
            if st.button(name, type="tertiary", key=f"preset_{name}"):
                _apply_preset(preset)


def _render_score_result(result: dict[str, Any] | None) -> None:
    with st.container(border=True):
        if result is None:
            st.markdown("#### :material/pending_actions: Awaiting a transaction")
            st.caption(
                "Score a transaction to see the decision, fraud score, and the "
                "SHAP drivers behind it."
            )
            return
        if "error" in result:
            st.error(
                result["error"],
                title="Scoring failed",
                icon=":material/error:",
            )
            return

        score = float(result["fraud_score"])
        threshold = float(result["threshold"])
        latency = float(result["latency_ms"])
        blocked = result["decision"] != DECISION_APPROVE

        st.markdown("#### Decision")
        decision_badge(result["decision"])
        st.metric("Fraud score", f"{score:.1%}")
        st.progress(
            min(max(score, 0.0), 1.0),
            text=f"Operating threshold {threshold:.0%} · inference {latency:.0f} ms",
        )
        st.caption(
            "Blocked decisions exceed the precision-weighted threshold. Red bars push "
            "the score up; green bars pull it down."
        )

        reasons = result.get("reasons", [])
        if reasons:
            st.echarts_chart(
                shap_option(reasons),
                height=210,
                alt="Top SHAP contributions for this transaction, red raising risk and green lowering it.",
            )
            render_reason_codes(result.get("reason_codes", []))
            with st.expander("Feature-level detail"):
                detail = pd.DataFrame(reasons).rename(
                    columns={
                        "feature": "Feature",
                        "value": "Value",
                        "shap": "SHAP",
                        "direction": "Direction",
                    }
                )
                st.dataframe(
                    detail,
                    column_config={
                        "Value": st.column_config.NumberColumn(format="%.4f"),
                        "SHAP": st.column_config.NumberColumn(format="%+.4f"),
                    },
                    hide_index=True,
                    alt="SHAP value and direction for each top risk driver.",
                )


def score_page() -> None:
    page_header(
        "Score a transaction",
        "Run a single transaction through the live model and inspect the decision drivers.",
        ":material/bolt:",
    )

    online, _ = api_health()
    if not online:
        st.warning(
            f"The scoring API at {API_URL} is offline. The form still works, but "
            "scoring needs the FastAPI service running.",
            title="Scoring API unavailable",
            icon=":material/cloud_off:",
        )
        with st.expander("Start the scoring API"):
            st.code("uvicorn api.main:app --port 8000", language="bash")

    mode = dataset_mode()
    _init_score_defaults(mode)

    left, right = st.columns([0.55, 0.45], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("**Transaction details**")
            st.caption(
                "PCA credit-card model (V1–V28)"
                if mode == "kaggle"
                else "Synthetic transaction schema"
            )
            _render_presets(KAGGLE_PRESETS if mode == "kaggle" else SYNTH_PRESETS)
            st.space("small")

            with st.form("score_form", border=False):
                if mode == "kaggle":
                    amount = st.number_input(
                        "Amount (USD)",
                        min_value=0.01,
                        max_value=100_000.0,
                        step=1.0,
                        key="score_amount",
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        v1 = st.slider("V1", -10.0, 10.0, step=0.1, key="score_V1")
                        v2 = st.slider("V2", -10.0, 10.0, step=0.1, key="score_V2")
                    with col2:
                        v3 = st.slider("V3", -10.0, 10.0, step=0.1, key="score_V3")
                        v4 = st.slider("V4", -10.0, 10.0, step=0.1, key="score_V4")
                    advanced: dict[str, float] = {}
                    with st.expander("Advanced components (V10–V27)"):
                        acol1, acol2 = st.columns(2)
                        split = (len(KAGGLE_ADVANCED) + 1) // 2
                        for name in KAGGLE_ADVANCED[:split]:
                            advanced[name] = acol1.slider(
                                name,
                                -10.0,
                                10.0,
                                step=0.1,
                                key=f"score_{name}",
                            )
                        for name in KAGGLE_ADVANCED[split:]:
                            advanced[name] = acol2.slider(
                                name,
                                -10.0,
                                10.0,
                                step=0.1,
                                key=f"score_{name}",
                            )
                else:
                    amount = st.number_input(
                        "Amount (USD)",
                        min_value=0.01,
                        max_value=1_000_000.0,
                        step=10.0,
                        key="score_amount",
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        tx_type = st.selectbox(
                            "Transaction type", TYPES, key="score_tx_type"
                        )
                        merchant = st.selectbox(
                            "Merchant category", MCC, key="score_merchant_category"
                        )
                        country = st.selectbox(
                            "Country", COUNTRIES, key="score_country"
                        )
                    with col2:
                        device = st.slider(
                            "Device risk", 0.0, 1.0, step=0.01, key="score_device_risk"
                        )
                        ip_risk = st.slider(
                            "IP risk", 0.0, 1.0, step=0.01, key="score_ip_risk"
                        )
                        account_age = st.number_input(
                            "Account age (days)",
                            min_value=0,
                            max_value=10_000,
                            step=30,
                            key="score_account_age_days",
                        )
                        tx_count = st.number_input(
                            "Transactions in last 24h",
                            min_value=0,
                            max_value=200,
                            step=1,
                            key="score_tx_count_24h",
                        )

                submitted = st.form_submit_button(
                    "Score transaction",
                    type="primary",
                    icon=":material/bolt:",
                    width="stretch",
                )

        if submitted:
            if mode == "kaggle":
                payload: dict[str, Any] = {"amount": float(amount)}
                payload.update(
                    {
                        "V1": float(v1),
                        "V2": float(v2),
                        "V3": float(v3),
                        "V4": float(v4),
                    }
                )
                payload.update({name: float(value) for name, value in advanced.items()})
            else:
                payload = {
                    "amount": float(amount),
                    "tx_type": tx_type,
                    "merchant_category": merchant,
                    "country": country,
                    "device_risk": float(device),
                    "ip_risk": float(ip_risk),
                    "account_age_days": int(account_age),
                    "tx_count_24h": int(tx_count),
                }

            with st.spinner("Scoring transaction..."):
                result, error = score_transaction(payload)
            if error:
                st.session_state["last_result"] = {"error": error}
            else:
                st.session_state["last_result"] = result
                history = st.session_state.setdefault("score_history", [])
                history.insert(
                    0,
                    {
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "amount": payload["amount"],
                        "score": float(result["fraud_score"]),
                        "decision": result["decision"],
                    },
                )
                st.session_state["score_history"] = history[:10]
                clear_prediction_cache()

    with right:
        _render_score_result(st.session_state.get("last_result"))
        history = st.session_state.get("score_history", [])
        if history:
            with st.expander(f"Session history ({len(history)})"):
                st.dataframe(
                    pd.DataFrame(history),
                    column_config={
                        "time": st.column_config.TextColumn("Time"),
                        "amount": st.column_config.NumberColumn(
                            "Amount", format="$%,.2f"
                        ),
                        "score": st.column_config.ProgressColumn(
                            "Fraud score",
                            min_value=0.0,
                            max_value=1.0,
                            format="percent",
                        ),
                        "decision": st.column_config.TextColumn("Decision"),
                    },
                    hide_index=True,
                    alt="Transactions scored in this browser session.",
                )


# ---------------------------------------------------------------------------
# Batch scoring page
# ---------------------------------------------------------------------------


def _schema_label(mode: str) -> str:
    return (
        "Kaggle PCA schema (V1–V28)"
        if mode == "kaggle"
        else "Transaction schema"
    )


def _model_label(mode: str) -> str:
    return "Kaggle PCA model" if mode == "kaggle" else "Transaction model"


def _execute_batch(
    prepared: pd.DataFrame,
    mode: str,
    source: str,
    name: str,
    signature: Any,
) -> bool:
    """Score a prepared frame and store the results in session state."""
    payloads = [
        upload_row_to_payload(row, mode) for _, row in prepared.iterrows()
    ]
    with st.spinner(f"Scoring {len(payloads):,} transactions..."):
        result, error = score_batch(payloads)
    if error:
        st.error(error, title="Batch scoring failed", icon=":material/error:")
        return False

    scored = prepared.copy()
    scored["fraud_score"] = [float(value) for value in result["scores"]]
    scored["decision"] = list(result["decisions"])

    if "_label" in scored.columns and scored["_label"].notna().any():
        labeled = scored["_label"].notna()
        actual = scored["_label"].fillna(-1).astype(int)
        flagged = (scored["decision"] == DECISION_BLOCK).astype(int)
        scored["actual_label"] = scored["_label"]
        scored["outcome"] = ""
        scored.loc[
            labeled & (actual == 1) & (flagged == 1), "outcome"
        ] = "True positive"
        scored.loc[
            labeled & (actual == 1) & (flagged == 0), "outcome"
        ] = "False negative"
        scored.loc[
            labeled & (actual == 0) & (flagged == 1), "outcome"
        ] = "False positive"
        scored.loc[
            labeled & (actual == 0) & (flagged == 0), "outcome"
        ] = "True negative"

    st.session_state["batch_results"] = {
        "source": source,
        "name": name,
        "signature": signature,
        "mode": mode,
        "threshold": float(result["threshold"]),
        "latency_ms": float(result["latency_ms"]),
        "scored": scored,
    }
    return True


def _render_batch_results(results: dict[str, Any]) -> None:
    scored = results["scored"]
    total = len(scored)
    flagged = int((scored["decision"] == DECISION_BLOCK).sum())
    threshold = float(results["threshold"])
    has_labels = (
        "actual_label" in scored.columns
        and scored["actual_label"].notna().any()
    )

    st.space("small")
    with st.container(horizontal=True):
        st.metric("Rows scored", f"{total:,}", border=True)
        st.metric(
            "Block / review",
            f"{flagged:,}",
            delta=f"{flagged / total:.1%}" if total else "0.0%",
            delta_description="of upload",
            border=True,
        )
        st.metric(
            "Mean fraud score",
            f"{scored['fraud_score'].mean():.1%}",
            border=True,
        )
        st.metric(
            "Max fraud score",
            f"{scored['fraud_score'].max():.1%}",
            border=True,
        )
        st.metric(
            "Batch inference",
            f"{results['latency_ms']:.0f} ms",
            border=True,
            help="Server-side time to score the whole file.",
        )

    st.space("small")
    left, right = st.columns([0.56, 0.44], gap="large")
    bands = score_band_counts(scored["fraud_score"], scored["decision"])

    with left:
        with st.container(border=True):
            st.markdown("**Score distribution**")
            st.caption(
                f"{results['name']} · {_schema_label(results['mode'])} · "
                f"threshold {threshold:.0%}"
            )
            st.echarts_chart(
                batch_band_option(bands),
                height=300,
                alt="Uploaded transactions grouped by score band, split into approved and blocked.",
            )

    with right:
        if has_labels:
            labeled = scored[scored["actual_label"].notna()]
            actual = labeled["actual_label"].astype(int)
            flagged_labeled = (labeled["decision"] == DECISION_BLOCK).astype(int)
            tp = int(((actual == 1) & (flagged_labeled == 1)).sum())
            fp = int(((actual == 0) & (flagged_labeled == 1)).sum())
            fn = int(((actual == 1) & (flagged_labeled == 0)).sum())
            tn = int(((actual == 0) & (flagged_labeled == 0)).sum())
            classes = set(actual.unique().tolist())
            with st.container(border=True):
                st.markdown("**Label quality**")
                st.caption(
                    f"{len(labeled):,} labeled rows · threshold {threshold:.0%}"
                )
                if classes == {0, 1}:
                    st.echarts_chart(
                        confusion_option(
                            {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
                        ),
                        height=250,
                        alt="Correct and incorrect decisions for the labeled upload, split by actual class.",
                    )
                    recall = tp / max(tp + fn, 1)
                    precision = tp / max(tp + fp, 1)
                    caught = float(
                        labeled.loc[
                            (actual == 1) & (flagged_labeled == 1), "_amount"
                        ].sum()
                    )
                    missed = float(
                        labeled.loc[
                            (actual == 1) & (flagged_labeled == 0), "_amount"
                        ].sum()
                    )
                    st.caption(
                        f"Recall {recall:.1%} · precision {precision:.1%} · "
                        f"caught {fmt_money(caught)} of "
                        f"{fmt_money(caught + missed)} fraud value"
                    )
                else:
                    st.markdown(
                        ":orange-badge[Single class] This file contains "
                        f"{'fraud' if 1 in classes else 'legitimate'} rows only, "
                        "so recall and precision need a mixed file."
                    )
                    with st.container(horizontal=True):
                        st.metric("True positives", tp, border=True)
                        st.metric("False positives", fp, border=True)
                        st.metric("False negatives", fn, border=True)
                        st.metric("True negatives", tn, border=True)
        else:
            with st.container(border=True):
                st.markdown("**Highest-risk transactions**")
                st.caption(
                    "No `Class` or `is_fraud` label column found, so the file "
                    "shows decisions only."
                )
                top = scored.nlargest(min(8, total), "fraud_score").copy()
                top["decision_badge"] = top["decision"].apply(lambda d: [d])
                st.dataframe(
                    top[["_amount", "fraud_score", "decision_badge"]],
                    column_config={
                        "_amount": st.column_config.NumberColumn(
                            "Amount", format="$%,.2f"
                        ),
                        "fraud_score": st.column_config.ProgressColumn(
                            "Fraud score",
                            min_value=0.0,
                            max_value=1.0,
                            format="percent",
                        ),
                        "decision_badge": st.column_config.MultiselectColumn(
                            "Decision",
                            options=[DECISION_APPROVE, DECISION_BLOCK],
                            color=["green", "red"],
                        ),
                    },
                    hide_index=True,
                    alt="Highest-scoring transactions in the uploaded file.",
                )

    st.space("small")
    table = scored.copy()
    table["decision_badge"] = table["decision"].apply(lambda d: [d])
    if "outcome" in table.columns:
        table["outcome_badge"] = table["outcome"].apply(
            lambda value: [value] if value else []
        )
        table["actual_display"] = table["actual_label"].map(
            {0: "Legitimate", 1: "Fraud"}
        )

    display_columns = []
    if "timestamp" in table.columns:
        display_columns.append("timestamp")
    display_columns.append("_amount")
    if results["mode"] == "synthetic":
        display_columns.extend(
            column
            for column in ("tx_type", "country")
            if column in table.columns
        )
    display_columns.extend(["fraud_score", "decision_badge"])
    if "actual_display" in table.columns:
        display_columns.extend(["actual_display", "outcome_badge"])

    export = scored.drop(
        columns=[
            column
            for column in ("_amount", "_label")
            if column in scored.columns
        ]
    )

    with st.container(border=True):
        header_left, header_right = st.columns(
            [0.62, 0.38], vertical_alignment="center"
        )
        with header_left:
            st.markdown("**Scored transactions**")
            st.caption(
                f"{results['name']} · {total:,} rows · highest score first · "
                "batch uploads are not added to the live feed"
            )
        with header_right:
            st.download_button(
                "Download results",
                data=export.to_csv(index=False).encode("utf-8"),
                file_name="sentinelguard_batch_scored.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
            )
        st.dataframe(
            table.sort_values("fraud_score", ascending=False)[display_columns],
            column_config={
                "timestamp": st.column_config.DatetimeColumn(
                    "Time", format="MMM DD · HH:mm"
                ),
                "_amount": st.column_config.NumberColumn(
                    "Amount", format="$%,.2f"
                ),
                "fraud_score": st.column_config.ProgressColumn(
                    "Fraud score",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent",
                ),
                "decision_badge": st.column_config.MultiselectColumn(
                    "Decision",
                    options=[DECISION_APPROVE, DECISION_BLOCK],
                    color=["green", "red"],
                ),
                "actual_display": st.column_config.TextColumn("Actual"),
                "outcome_badge": st.column_config.MultiselectColumn(
                    "Outcome",
                    options=[
                        "True positive",
                        "False positive",
                        "False negative",
                        "True negative",
                    ],
                    color=["green", "orange", "red", "gray"],
                ),
            },
            hide_index=True,
            alt="Uploaded transactions with model score, decision, and outcome.",
        )

    if st.button(
        "Clear results",
        type="tertiary",
        icon=":material/close:",
        key="batch_clear",
    ):
        st.session_state.pop("batch_results", None)
        st.rerun()


def _score_sample() -> None:
    mode = dataset_mode()
    sample = load_traffic_sample(50, seed=42)
    if sample.empty:
        st.error(
            "The raw dataset needed for a sample was not found in `data/`.",
            title="Sample unavailable",
            icon=":material/folder_off:",
        )
        return
    prepared, _, errors = prepare_upload(sample, mode)
    if errors:
        st.error(
            " ".join(errors),
            title="Sample unavailable",
            icon=":material/error:",
        )
        return
    if _execute_batch(
        prepared,
        mode,
        "sample",
        f"{_schema_label(mode)} · 50-row sample",
        "sample",
    ):
        st.rerun()


def batch_page() -> None:
    page_header(
        "Batch scoring",
        "Upload a CSV of transactions, score the whole file in one request, and export the decisions.",
        ":material/upload_file:",
    )

    online, _ = api_health()
    if not online:
        st.warning(
            f"The scoring API at {API_URL} is offline. You can still upload and "
            "validate a file, but scoring needs the FastAPI service.",
            title="Scoring API unavailable",
            icon=":material/cloud_off:",
        )
        with st.expander("Start the scoring API"):
            st.code("uvicorn api.main:app --port 8000", language="bash")

    upload = st.file_uploader(
        "Transaction CSV",
        type=["csv"],
        key="batch_file",
        help=(
            "Kaggle PCA schema (Amount + V1–V28) or the synthetic transaction "
            f"schema. Up to {BATCH_MAX_ROWS:,} rows."
        ),
    )
    st.caption(
        "Both models are deployed: each file is scored by the model trained on "
        f"its schema. Metrics currently show the {_schema_label(dataset_mode())} model."
    )

    with st.expander("CSV format and templates"):
        st.markdown(
            "**Kaggle PCA schema** — `Amount` plus `V1`–`V28`. Add `Class` "
            "(0/1) to get recall, precision, and a confusion matrix.\n\n"
            "**Transaction schema** — `timestamp`, `amount`, `tx_type`, "
            "`merchant_category`, `country`, `device_risk`, `ip_risk`, "
            "`account_age_days`, `tx_count_24h`. Add `is_fraud` (0/1) for "
            "quality metrics.\n\n"
            "Both schemas are deployed. Each upload is scored by the model "
            "trained on the schema it uses; the matching template is the "
            "quickest way to start."
        )
        template_left, template_right = st.columns(2)
        with template_left:
            st.download_button(
                "Download Kaggle template",
                data=batch_template("kaggle").to_csv(index=False).encode("utf-8"),
                file_name="sentinelguard_kaggle_template.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
            )
        with template_right:
            st.download_button(
                "Download transaction template",
                data=batch_template("synthetic")
                .to_csv(index=False)
                .encode("utf-8"),
                file_name="sentinelguard_transaction_template.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
            )

    if upload is None:
        results = st.session_state.get("batch_results")
        if results and results.get("source") == "sample":
            _render_batch_results(results)
            return
        with st.container(border=True):
            st.markdown("#### :material/upload_file: Drop a CSV to begin")
            st.caption(
                "Preview, validate, and score the file in one pass. Use the "
                "templates above if you need a starting point."
            )
            if st.button(
                "Score 50 sample transactions",
                icon=":material/science:",
                disabled=not online,
                key="batch_sample",
            ):
                _score_sample()
        return

    upload.seek(0)
    try:
        raw = pd.read_csv(upload, nrows=BATCH_MAX_ROWS + 1)
    except Exception as exc:
        st.error(
            f"Could not read this file as CSV ({type(exc).__name__}: {exc}).",
            title="Unreadable file",
            icon=":material/error:",
        )
        return
    if raw.empty:
        st.error(
            "The file contains headers but no data rows.",
            title="Empty file",
            icon=":material/table_rows:",
        )
        return
    if len(raw) > BATCH_MAX_ROWS:
        raw = raw.head(BATCH_MAX_ROWS)
        st.warning(
            f"Only the first {BATCH_MAX_ROWS:,} rows will be scored.",
            icon=":material/content_cut:",
        )
    raw.columns = [str(column).strip() for column in raw.columns]

    mode = detect_upload_schema(raw)
    if mode is None:
        st.error(
            "Could not recognise the columns. Use the Kaggle PCA schema or the "
            "transaction schema shown above.",
            title="Unknown CSV schema",
            icon=":material/schema:",
        )
        return
    prepared, issues, errors = prepare_upload(raw, mode)
    if errors:
        st.error(
            "\n\n".join(errors),
            title="Cannot score this file",
            icon=":material/error:",
        )
        return
    if prepared.empty:
        st.error(
            "No rows are left after validation. Check the amount, risk, and "
            "account fields.",
            title="Nothing to score",
            icon=":material/filter_alt_off:",
        )
        return

    signature = (upload.name, upload.size)
    results = st.session_state.get("batch_results")
    if (
        results
        and results.get("source") == "upload"
        and results.get("signature") == signature
    ):
        _render_batch_results(results)
        return

    with st.container(border=True):
        st.markdown("**File check**")
        st.caption(
            f"{upload.name} · {len(prepared):,} scorable rows · "
            f"{_schema_label(mode)}"
        )
        preview = prepared.head(8).drop(
            columns=[
                column
                for column in ("_amount", "_label")
                if column in prepared.columns
            ]
        )
        st.dataframe(
            preview,
            hide_index=True,
            alt="First rows of the uploaded CSV after validation.",
        )
        if issues:
            st.caption("Data notes")
            for issue in issues:
                st.markdown(f"- {issue}")

    st.space("small")
    if st.button(
        f"Score {len(prepared):,} transactions",
        type="primary",
        icon=":material/bolt:",
        disabled=not online,
        key="batch_score",
    ):
        if _execute_batch(prepared, mode, "upload", upload.name, signature):
            st.rerun()
    st.caption(
        "Batch inference runs in one request and does not write to the live "
        "decision feed."
    )


# ---------------------------------------------------------------------------
# Model performance page
# ---------------------------------------------------------------------------


def performance_page() -> None:
    page_header(
        "Model performance",
        f"Held-out evaluation of the {_schema_label(dataset_mode())} RandomForest "
        "and its anomaly signal.",
        ":material/monitoring:",
    )

    metrics = load_metrics()
    if not metrics:
        empty_card(
            "No evaluation artifacts",
            "Run the training pipeline to create models/metrics.json.",
            ":material/folder_off:",
        )
        return

    recall = float(metrics.get("recall", 0.0))
    precision = float(metrics.get("precision", 0.0))
    fpr = float(metrics.get("fpr", 0.0))
    pr_auc = float(metrics.get("pr_auc", 0.0))
    roc_auc = float(metrics.get("roc_auc", 0.0))
    threshold = float(metrics.get("threshold", 0.0))
    brier = metrics.get("brier")

    with st.container(horizontal=True):
        st.metric(
            "Recall",
            f"{recall:.1%}",
            delta=f"{recall - 0.80:+.1%}",
            delta_description="vs 80% target",
            border=True,
            help="Fraud caught at the operating threshold.",
        )
        st.metric(
            "Precision",
            f"{precision:.1%}",
            border=True,
            help="Share of alerts that are real fraud.",
        )
        st.metric(
            "Accuracy",
            f"{float(metrics.get('accuracy', 0.0)):.2%}",
            border=True,
            help="Share of all held-out transactions classified correctly.",
        )
        st.metric(
            "False positive rate",
            f"{fpr:.3%}",
            border=True,
            help="Legitimate transactions incorrectly flagged.",
        )
        st.metric("PR-AUC", f"{pr_auc:.3f}", border=True, help="Imbalanced-data quality.")
        st.metric("ROC-AUC", f"{roc_auc:.3f}", border=True, help="Ranking quality.")
        st.metric(
            "Threshold",
            f"{threshold:.0%}",
            border=True,
            help="Precision-weighted score cutoff.",
        )
        if brier is not None:
            st.metric(
                "Brier score",
                f"{float(brier):.4f}",
                border=True,
                help="Mean squared error of predicted probabilities; lower is better.",
            )

    with st.spinner("Scoring the held-out test split..."):
        ev = load_eval_analytics()

    st.space("small")
    if ev:
        left, right = st.columns(2, gap="large")
        with left:
            with st.container(border=True):
                st.markdown("**ROC curve**")
                st.caption(f"AUC {roc_auc:.3f} · {ev['n_test']:,} held-out transactions")
                st.echarts_chart(
                    roc_option(ev),
                    height=320,
                    alt=f"ROC curve for the fraud model with AUC {roc_auc:.3f}.",
                )
        with right:
            with st.container(border=True):
                st.markdown("**Precision–recall curve**")
                st.caption(f"PR-AUC {pr_auc:.3f} · {ev['n_fraud']:,} fraud cases")
                st.echarts_chart(
                    pr_option(ev),
                    height=320,
                    alt=f"Precision-recall curve with PR-AUC {pr_auc:.3f}.",
                )

        st.space("small")
        left, right = st.columns([0.58, 0.42], gap="large")
        with left:
            with st.container(border=True):
                st.markdown("**Score distribution by class**")
                st.caption("Percent of each class within each score band")
                st.echarts_chart(
                    risk_band_option(ev),
                    height=320,
                    alt="Legitimate transactions concentrate below 5% while fraud concentrates above 20%.",
                )
        with right:
            cm = ev["confusion"]
            with st.container(border=True):
                st.markdown("**Confusion matrix**")
                st.caption(
                    f"At {threshold:.0%} · TN {cm['tn']:,} · FP {cm['fp']:,} · "
                    f"FN {cm['fn']:,} · TP {cm['tp']:,}"
                )
                st.echarts_chart(
                    confusion_option(cm),
                    height=320,
                    alt="Classification outcomes split by actual class at the operating threshold.",
                )

        st.space("small")
        calibration = metrics.get("calibration") or {}
        curve = [row for row in calibration.get("curve", []) if row.get("count")]
        baselines = metrics.get("baselines") or {}
        left, right = st.columns([0.55, 0.45], gap="large")
        with left:
            with st.container(border=True):
                st.markdown("**Probability calibration**")
                if curve:
                    st.caption(
                        f"Brier {float(calibration.get('brier', 0.0)):.4f} · "
                        "points on the dashed line mean the score matches the observed fraud rate"
                    )
                    st.echarts_chart(
                        calibration_option(curve),
                        height=300,
                        alt="Reliability curve comparing predicted fraud probability with observed fraud rate.",
                    )
                else:
                    st.caption(
                        "Calibration is unavailable. Rerun training to write it into "
                        "models/metrics.json."
                    )
        with right:
            with st.container(border=True):
                st.markdown("**Baseline comparison**")
                baseline_frames = []
                for key in ("majority", "flag_all", "random"):
                    info = baselines.get(key)
                    if not info:
                        continue
                    baseline_frames.append(
                        {
                            "strategy": info.get("description", key),
                            "precision": float(info.get("precision", 0.0)),
                            "recall": float(info.get("recall", 0.0)),
                            "flagged_rate": float(info.get("flagged_rate", 0.0)),
                        }
                    )
                baseline_frames.append(
                    {
                        "strategy": "Model at operating threshold",
                        "precision": cm["precision"],
                        "recall": cm["recall"],
                        "flagged_rate": cm["alerts"] / max(ev["n_test"], 1),
                    }
                )
                st.dataframe(
                    pd.DataFrame(baseline_frames),
                    column_config={
                        "strategy": st.column_config.TextColumn("Strategy"),
                        "precision": st.column_config.NumberColumn(
                            "Precision", format="percent"
                        ),
                        "recall": st.column_config.NumberColumn(
                            "Recall", format="percent"
                        ),
                        "flagged_rate": st.column_config.NumberColumn(
                            "Flagged", format="percent"
                        ),
                    },
                    hide_index=True,
                    alt="Model quality next to trivial baseline strategies.",
                )
                st.caption(
                    f"Fraud prevalence on the held-out split: "
                    f"{float(baselines.get('prevalence', 0.0)):.2%}"
                )

        st.space("small")
        policy = load_policy()
        rows = policy_rows(policy)
        with st.container(border=True):
            header_left, header_right = st.columns([0.62, 0.38], vertical_alignment="center")
            with header_left:
                st.markdown("**Threshold policy simulation**")
                st.caption(
                    "Cutoffs that trade review cost against missed fraud. "
                    "Decision support, not approved rules."
                )
            with header_right:
                if rows:
                    st.download_button(
                        "Download policy CSV",
                        data=pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
                        file_name="sentinelguard_threshold_policy.csv",
                        mime="text/csv",
                        icon=":material/download:",
                        width="stretch",
                    )
            if rows:
                table = pd.DataFrame(rows)
                table["policy_badge"] = table["policy"].apply(
                    lambda name: [name.replace("_", " ").title()]
                )
                st.dataframe(
                    table[["policy_badge", "purpose", "threshold", "precision",
                           "recall", "fpr", "flagged_rate", "alerts", "cost"]],
                    column_config={
                        "policy_badge": st.column_config.MultiselectColumn(
                            "Policy",
                            options=[
                                name.replace("_", " ").title()
                                for name in table["policy"]
                            ],
                            color=["blue"] * len(table),
                        ),
                        "purpose": st.column_config.TextColumn("Purpose"),
                        "threshold": st.column_config.NumberColumn(
                            "Cutoff", format="%.3f"
                        ),
                        "precision": st.column_config.ProgressColumn(
                            "Precision", min_value=0.0, max_value=1.0, format="percent"
                        ),
                        "recall": st.column_config.ProgressColumn(
                            "Recall", min_value=0.0, max_value=1.0, format="percent"
                        ),
                        "fpr": st.column_config.NumberColumn("FPR", format="percent"),
                        "flagged_rate": st.column_config.NumberColumn(
                            "Flagged", format="percent"
                        ),
                        "alerts": st.column_config.NumberColumn("Alerts", format="%,d"),
                        "cost": st.column_config.NumberColumn(
                            "Expected cost", format="$%,.0f"
                        ),
                    },
                    hide_index=True,
                    alt="Threshold policy candidates with precision, recall, review load and expected cost.",
                )
                assumptions = policy.get("assumptions", {})
                st.caption(
                    f"Assumes ${float(assumptions.get('review_cost', 0)):,.2f} per review and "
                    f"${float(assumptions.get('fraud_handling_cost', 0)):,.2f} per missed fraud, "
                    f"with a {float(assumptions.get('review_capacity', 0)):.0%} review budget."
                )
            else:
                st.caption(
                    "No policy artifact found. Run `python -m src.policy --data "
                    "data/transactions.csv` to generate one."
                )

        st.space("small")
        left, right = st.columns([0.55, 0.45], gap="large")
        with left:
            with st.container(border=True):
                st.markdown("**Business impact**")
                st.caption("Amounts summed over the held-out test split at the operating threshold.")
                with st.container(horizontal=True):
                    st.metric(
                        "Fraud value caught",
                        fmt_money(ev["business"]["caught_amount"]),
                        border=True,
                    )
                    st.metric(
                        "Fraud value missed",
                        fmt_money(ev["business"]["missed_amount"]),
                        border=True,
                    )
                    st.metric(
                        "Value recovery",
                        f"{ev['business']['recovery_rate']:.1%}",
                        border=True,
                    )
                st.caption(
                    f"{cm['alerts']:,} alerts raised · "
                    f"{cm['alerts_per_10k']:.1f} false alerts per 10k legitimate transactions · "
                    f"specificity {cm['specificity']:.3%}"
                )
        with right:
            with st.container(border=True):
                st.markdown("**Operating configuration**")
                if metrics.get("targets_met"):
                    st.markdown(":green-badge[Targets met] Recall > 80% and FPR < 5%")
                else:
                    st.markdown(
                        ":orange-badge[Targets partially met] Recall target is 80%"
                    )
                params = metrics.get("best_params", {})
                if params:
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Parameter": [
                                    "Trees",
                                    "Max depth",
                                    "Max features",
                                    "Min samples leaf",
                                    "Class weight",
                                ],
                                "Value": [
                                    str(params.get("n_estimators", "—")),
                                    str(params.get("max_depth") or "unlimited"),
                                    str(params.get("max_features", "—")),
                                    str(params.get("min_samples_leaf", "—")),
                                    str(params.get("class_weight", "—")),
                                ],
                            }
                        ),
                        column_config={"Value": st.column_config.TextColumn("Value")},
                        hide_index=True,
                        alt="RandomForest hyperparameters.",
                    )
                st.caption(
                    "Class-balanced RandomForest with an Isolation Forest anomaly "
                    "feature and a precision-weighted threshold search."
                )
    else:
        empty_card(
            "Evaluation analytics unavailable",
            "The bundle or raw dataset could not be loaded. Check models/bundle.joblib "
            "and data/creditcard.csv (or data/transactions.csv).",
            ":material/analytics:",
        )

    shap_path = shap_image_path()
    if shap_path.exists():
        st.space("small")
        with st.container(border=True):
            st.markdown("**Global feature importance**")
            st.caption(
                "Mean absolute SHAP value across a sample of the test split, "
                "generated at training time."
            )
            st.image(
                str(shap_path),
                width="stretch",
                caption="SHAP summary: features pushing transactions toward fraud.",
                alt="SHAP summary plot ranking the most influential fraud features.",
            )


# ---------------------------------------------------------------------------
# Drift monitor page
# ---------------------------------------------------------------------------


def _drift_guide() -> None:
    with st.container(border=True):
        st.markdown("**How to read PSI**")
        st.caption(
            "Population Stability Index compares the training reference with recent "
            "production scores. Lower is better."
        )
        st.caption(
            "PSI is noisy on small windows: use at least a few hundred recent "
            "predictions before acting on a warning."
        )
        st.markdown(
            ":green-badge[Stable] PSI < 0.10 — no meaningful shift\n\n"
            ":orange-badge[Warning] 0.10 ≤ PSI < 0.25 — monitor and investigate\n\n"
            ":red-badge[Alert] PSI ≥ 0.25 — retraining is recommended"
        )


def drift_page() -> None:
    page_header(
        "Drift monitor",
        f"Population Stability Index between the {_model_label(dataset_mode())} "
        "reference and recent production traffic.",
        ":material/sensors:",
    )

    online, _ = api_health()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        window = st.selectbox(
            "Prediction window",
            [200, 500, 1000, 2000],
            index=1,
            key="drift_window",
            help="Most recent logged predictions compared with the reference sample.",
        )
        run_check = st.button(
            "Run drift check",
            type="primary",
            icon=":material/play_arrow:",
            disabled=not online,
        )
        if not online:
            st.caption("Start the API to run a live drift check.")

    if run_check:
        with st.spinner("Computing PSI across monitored features..."):
            report, error = fetch_drift(window, dataset_mode())
        st.session_state["drift_report"] = report
        st.session_state["drift_error"] = error
        st.session_state["drift_checked_at"] = datetime.now()
        st.session_state["drift_window_used"] = window

    error = st.session_state.get("drift_error")
    report = st.session_state.get("drift_report")
    checked_at = st.session_state.get("drift_checked_at")

    if error:
        st.error(error, title="Drift check failed", icon=":material/error:")

    if not report:
        st.space("small")
        with st.container(border=True):
            st.markdown("**No drift check yet**")
            st.caption(
                "Run a check to compare recent production scores with the training "
                "reference, or generate demo traffic to create a comparison window."
            )
            if st.button(
                "Generate 60 demo transactions",
                icon=":material/science:",
                disabled=not online,
                key="drift_demo_empty",
            ):
                simulate_traffic(60)
        st.space("small")
        _drift_guide()
        return

    if "message" in report:
        st.info(
            report["message"],
            title="Not enough production traffic",
            icon=":material/hourglass_empty:",
        )
        st.caption(
            "Drift needs at least 50 logged predictions. Generate demo traffic to "
            "exercise the full monitoring path, or send transactions through the API."
        )
        if st.button(
            "Generate 60 demo transactions",
            icon=":material/science:",
            disabled=not online,
            key="drift_demo_insufficient",
        ):
            simulate_traffic(60)
        st.space("small")
        _drift_guide()
        return

    rows = drift_rows(report)
    if not rows:
        st.info("The drift report did not contain any monitored features.")
        return

    alerts = sum(1 for row in rows if row["status"] == "ALERT")
    warnings = sum(1 for row in rows if row["status"] == "WARN")
    max_psi = max(row["psi"] for row in rows)

    with st.container(horizontal=True):
        st.metric("Features monitored", len(rows), border=True)
        st.metric("Alerts", alerts, border=True, help="PSI ≥ 0.25")
        st.metric("Warnings", warnings, border=True, help="0.10 ≤ PSI < 0.25")
        st.metric("Max PSI", f"{max_psi:.3f}", border=True)

    if report.get("_retrain_recommended"):
        st.warning(
            "At least one feature crossed the 0.25 alert threshold. Investigate the "
            "upstream data change and schedule retraining.",
            title="Retraining recommended",
            icon=":material/model_training:",
        )
    else:
        st.success(
            "No feature crossed the alert threshold. Keep monitoring the window.",
            title="Distribution is stable",
            icon=":material/check_circle:",
        )

    st.space("small")
    left, right = st.columns([0.55, 0.45], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**PSI by feature**")
            st.caption("Alert threshold 0.25 · warning threshold 0.10")
            st.echarts_chart(
                psi_option(rows),
                height=max(260, 34 * len(rows)),
                alt="PSI by monitored feature with warning and alert thresholds.",
            )
    with right:
        with st.container(border=True):
            st.markdown("**Feature detail**")
            table = pd.DataFrame(rows)
            table["status_badge"] = table["status"].apply(
                lambda status: [STATUS_LABELS.get(status, status)]
            )
            st.dataframe(
                table,
                column_config={
                    "feature": st.column_config.TextColumn("Feature"),
                    "psi": st.column_config.NumberColumn("PSI", format="%.4f"),
                    "status": None,
                    "status_badge": st.column_config.MultiselectColumn(
                        "Status",
                        options=["Stable", "Warning", "Alert"],
                        color=["green", "orange", "red"],
                    ),
                },
                hide_index=True,
                alt="PSI value and status for every monitored feature.",
            )
            if checked_at:
                st.caption(
                    f"Checked {checked_at:%d %b %Y · %H:%M} · "
                    f"last {st.session_state.get('drift_window_used', window):,} predictions"
                )
            if st.button(
                "Generate 60 demo transactions",
                icon=":material/science:",
                disabled=not online,
                key="drift_demo_populated",
            ):
                simulate_traffic(60)

    st.space("small")
    _drift_guide()


# ---------------------------------------------------------------------------
# Live activity page
# ---------------------------------------------------------------------------


def activity_page() -> None:
    page_header(
        "Live activity",
        "Every transaction scored by the API, newest first, with filtering and export.",
        ":material/receipt_long:",
    )

    predictions = load_predictions()
    if predictions.empty:
        with st.container(border=True):
            st.markdown("#### :material/receipt_long: No decisions logged yet")
            st.caption(
                "Score a transaction from the console or generate demo traffic to "
                "populate the feed."
            )
            left, right = st.columns([0.5, 0.5], vertical_alignment="center")
            with left:
                st.page_link(
                    PAGES["score"],
                    label="Open scoring console",
                    icon=":material/bolt:",
                )
            with right:
                if st.button(
                    "Generate 60 demo transactions",
                    icon=":material/science:",
                    disabled=not api_health()[0],
                    key="activity_demo",
                ):
                    simulate_traffic(60)
        return

    with st.container(horizontal=True, vertical_alignment="bottom"):
        decision_filter = st.segmented_control(
            "Decision",
            ["All", "Approved", "Block / review"],
            default="All",
            key="activity_decision",
        )
        min_score = st.slider(
            "Minimum score", 0.0, 1.0, 0.0, 0.01, key="activity_min_score"
        )
        window = st.selectbox(
            "Window",
            ["All", "Last 50", "Last 200", "Last 1000"],
            index=0,
            key="activity_window",
        )

    filtered = predictions.copy()
    if decision_filter == "Approved":
        filtered = filtered[filtered["decision"] == DECISION_APPROVE]
    elif decision_filter == "Block / review":
        filtered = filtered[filtered["decision"] == DECISION_BLOCK]
    if "score" in filtered.columns:
        filtered = filtered[filtered["score"] >= min_score]
    if window != "All":
        filtered = filtered.tail(int(window.split()[-1]))

    if filtered.empty:
        st.info("No decisions match the current filters.", icon=":material/filter_alt_off:")
        return

    blocked = int((filtered["decision"] == DECISION_BLOCK).sum())
    with st.container(horizontal=True):
        st.metric("Matching decisions", f"{len(filtered):,}", border=True)
        st.metric(
            "Block / review",
            f"{blocked:,}",
            delta=f"{blocked / len(filtered):.1%}",
            delta_description="of filtered",
            border=True,
        )
        st.metric(
            "Mean score",
            f"{filtered['score'].mean():.1%}",
            border=True,
        )
        st.metric(
            "Max score",
            f"{filtered['score'].max():.1%}",
            border=True,
        )
        st.metric(
            "Last decision",
            f"{filtered['timestamp'].max():%H:%M:%S}",
            border=True,
        )

    st.space("small")
    with st.container(border=True):
        st.markdown("**Score trend**")
        st.caption("Fraud score over time · dashed line is the operating threshold")
        trend = filtered.dropna(subset=["timestamp", "score"]).sort_values("timestamp")
        if len(trend) >= 2:
            st.echarts_chart(
                score_trend_option(trend, float(load_metrics().get("threshold", 0.0))),
                height=260,
                alt="Fraud score over time with the operating threshold marked.",
            )
        else:
            st.caption("At least two decisions are needed to draw a trend.")

    st.space("small")
    with st.container(border=True):
        header_left, header_right = st.columns([0.7, 0.3], vertical_alignment="center")
        with header_left:
            st.markdown("**Decision log**")
        with header_right:
            st.download_button(
                "Export CSV",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="sentinelguard_decisions.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
            )

        table = filtered.iloc[::-1].copy()
        table["decision_badge"] = table["decision"].apply(lambda d: [d])
        columns = ["timestamp", "amount", "score", "decision_badge"]
        if "anomaly" in table.columns:
            columns.append("anomaly")
        st.dataframe(
            table[columns],
            column_config={
                "timestamp": st.column_config.DatetimeColumn(
                    "Time", format="MMM DD · HH:mm:ss"
                ),
                "amount": st.column_config.NumberColumn("Amount", format="$%,.2f"),
                "score": st.column_config.ProgressColumn(
                    "Fraud score",
                    min_value=0.0,
                    max_value=1.0,
                    format="percent",
                ),
                "decision_badge": st.column_config.MultiselectColumn(
                    "Decision",
                    options=[DECISION_APPROVE, DECISION_BLOCK],
                    color=["green", "red"],
                ),
                "anomaly": st.column_config.NumberColumn(
                    "Anomaly score", format="%.3f"
                ),
            },
            hide_index=True,
            alt="Filtered decision log with time, amount, fraud score, and decision.",
        )
        st.caption(f"{len(filtered):,} decisions · newest first")


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


PAGES = {
    "overview": st.Page(
        overview_page,
        title="Overview",
        icon=":material/space_dashboard:",
        url_path="overview",
        default=True,
    ),
    "activity": st.Page(
        activity_page,
        title="Live activity",
        icon=":material/receipt_long:",
        url_path="activity",
    ),
    "score": st.Page(
        score_page,
        title="Score transaction",
        icon=":material/bolt:",
        url_path="score",
    ),
    "batch": st.Page(
        batch_page,
        title="Batch scoring",
        icon=":material/upload_file:",
        url_path="batch",
    ),
    "performance": st.Page(
        performance_page,
        title="Model performance",
        icon=":material/monitoring:",
        url_path="performance",
    ),
    "drift": st.Page(
        drift_page,
        title="Drift monitor",
        icon=":material/sensors:",
        url_path="drift",
    ),
}


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### :material/shield: SentinelGuard")
        st.caption("Real-time fraud intelligence")

        online, _ = api_health()
        if online:
            st.badge("Scoring API online", icon=":material/check_circle:", color="green")
        else:
            st.badge("Scoring API offline", icon=":material/error:", color="red")
        st.caption(f"`{API_URL}`")

        st.space("small")
        st.radio(
            "Active model",
            MODEL_SCHEMAS,
            format_func=_model_label,
            key="model_schema",
            help=(
                "Switches the metrics, scoring console, and drift reference. "
                "Batch uploads are always routed by the file's schema."
            ),
        )

        st.space("medium")
        metrics = load_metrics()
        with st.container(border=True):
            st.caption("Model")
            st.markdown("**RandomForest · class-balanced**")
            st.caption(_model_label(dataset_mode()))
            st.caption("Isolation Forest anomaly signal")
            st.markdown(f"Threshold `{float(metrics.get('threshold', 0)):.0%}`")
            st.caption(f"Artifacts updated {artifact_age()}")

        st.space("small")
        st.link_button(
            "API reference",
            f"{API_URL}/docs",
            icon=":material/open_in_new:",
            width="stretch",
            disabled=not online,
        )
        st.caption("SentinelGuard v1.0 · internship build")


def main() -> None:
    navigation = st.navigation(
        {
            "Monitor": [PAGES["overview"], PAGES["activity"]],
            "Decide": [PAGES["score"], PAGES["batch"]],
            "Govern": [PAGES["performance"], PAGES["drift"]],
        }
    )
    render_sidebar()
    navigation.run()


if __name__ == "__main__":
    main()
