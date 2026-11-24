"""Generate a synthetic, highly imbalanced transaction log (~1% fraud).
Columns match the portfolio PDF: amount, timestamp, type, merchant category,
country, device risk score, IP risk score.
Usage: python -m src.make_data --n 300000
"""
import argparse
import numpy as np
import pandas as pd
from src.features import TYPES, MCC, COUNTRIES


def make(n=300_000, fraud_rate=0.01, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    secs = np.sort(rng.integers(0, 120 * 24 * 3600, n))
    ts = pd.Timestamp("2026-01-01") + pd.to_timedelta(secs, unit="s")
    amount = np.round(rng.lognormal(4.0, 1.2, n), 2)
    tx_type = rng.choice(TYPES, n, p=[0.55, 0.15, 0.10, 0.20])
    mcc = rng.choice(MCC, n, p=[.2, .1, .1, .08, .15, .03, .02, .15, .1, .07])
    country = rng.choice(COUNTRIES, n, p=[.86, .04, .03, .02, .02, .01, .01, .01])
    device = rng.beta(1.2, 6, n)
    ip = rng.beta(1.2, 6, n)
    age = rng.integers(5, 3000, n)
    c24 = rng.poisson(2.0, n)
    hour = ts.hour.values

    z = (0.8 * (np.log1p(amount) - 4.5) + 3.0 * device + 3.0 * ip
         + 1.5 * (country != "IN") + 1.2 * (hour < 6) + 1.0 * (tx_type == "transfer")
         + 1.0 * np.isin(mcc, ["gambling", "crypto"]) - age / 1500 + 0.2 * c24)
    score = z + rng.gumbel(0, 1.0, n)  # noise so the problem is not trivially separable
    label = (score >= np.quantile(score, 1 - fraud_rate)).astype(int)

    return pd.DataFrame({
        "timestamp": ts, "amount": amount, "tx_type": tx_type,
        "merchant_category": mcc, "country": country,
        "device_risk": device.round(4), "ip_risk": ip.round(4),
        "account_age_days": age, "tx_count_24h": c24, "is_fraud": label,
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300_000)
    ap.add_argument("--out", default="data/transactions.csv")
    a = ap.parse_args()
    df = make(a.n)
    df.to_csv(a.out, index=False)
    print(f"saved {len(df):,} rows, fraud rate {df.is_fraud.mean():.3%} -> {a.out}")
