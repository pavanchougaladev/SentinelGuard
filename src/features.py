"""Shared feature engineering. Used by training AND the API so they never drift apart."""
import numpy as np
import pandas as pd

TYPES = ["purchase", "transfer", "withdrawal", "online"]
MCC = ["grocery", "fuel", "electronics", "travel", "dining",
       "gambling", "crypto", "utilities", "clothing", "other"]
COUNTRIES = ["IN", "US", "GB", "AE", "SG", "NG", "RU", "CN"]
HOME = "IN"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Support Kaggle Credit Card dataset (V1-V28 + Amount)
    if any(col.startswith("V") and col[1:].isdigit() for col in df.columns):
        X = pd.DataFrame(index=df.index)
        v_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in df.columns]
        for col in v_cols:
            X[col] = df[col].astype(float)
        amt_col = "Amount" if "Amount" in df.columns else ("amount" if "amount" in df.columns else None)
        if amt_col:
            X["log_amount"] = np.log1p(df[amt_col].astype(float))
        elif "log_amount" in df.columns:
            X["log_amount"] = df["log_amount"].astype(float)
        else:
            X["log_amount"] = 0.0
        return X

    X = pd.DataFrame(index=df.index)
    ts = pd.to_datetime(df["timestamp"])
    X["log_amount"] = np.log1p(df["amount"].astype(float))
    X["device_risk"] = df["device_risk"].astype(float)
    X["ip_risk"] = df["ip_risk"].astype(float)
    X["account_age_days"] = df["account_age_days"].astype(float)
    X["tx_count_24h"] = df["tx_count_24h"].astype(float)
    X["hour"] = ts.dt.hour.astype(float)
    X["is_night"] = ts.dt.hour.isin([0, 1, 2, 3, 4, 5]).astype(float)
    X["dow"] = ts.dt.dayofweek.astype(float)
    X["is_foreign"] = (df["country"] != HOME).astype(float)
    X["risk_product"] = X["device_risk"] * X["ip_risk"]
    for t in TYPES:
        X[f"type_{t}"] = (df["tx_type"] == t).astype(float)
    for m in MCC:
        X[f"mcc_{m}"] = (df["merchant_category"] == m).astype(float)
    for c in COUNTRIES:
        X[f"country_{c}"] = (df["country"] == c).astype(float)
    return X
