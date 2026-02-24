# Sentinel AI — anomaly (S2-01)
# Isolation Forest anomaly detectors: one model per country on weekly GDELT aggregates.
# See GitHub Issue #15.

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ANOMALY_FEATURES = [
    "goldstein_mean",
    "goldstein_std",
    "goldstein_min",
    "mentions_total",
    "avg_tone",
    "event_count",
]

COUNTRIES = ["UA", "TW", "IR", "VE", "PK", "ET", "RS", "BR"]


def _repo_root() -> Path:
    """Repo root (HackathonPractice)."""
    return Path(__file__).resolve().parents[2]


def _models_dir() -> Path:
    """models/ at repo root."""
    d = _repo_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _gdelt_path(country_code: str) -> Path:
    """Path to data/gdelt/{CC}_events.csv."""
    return _repo_root() / "data" / "gdelt" / f"{country_code}_events.csv"


def train_anomaly_detector(country_code: str, gdelt_df: pd.DataFrame):
    """
    Train one Isolation Forest per country.
    Learns country-specific 'normal' baseline from weekly aggregates.
    Saves models/anomaly_{CC}.pkl and models/scaler_{CC}.pkl.
    """
    df = gdelt_df.copy()
    df["date"] = pd.to_datetime(
        df["SQLDATE"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df = df.dropna(subset=["date"])
    for col in ["GoldsteinScale", "NumMentions", "AvgTone"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["week"] = df["date"].dt.to_period("W")
    weekly = (
        df.groupby("week")
        .agg(
            {
                "GoldsteinScale": ["mean", "std", "min"],
                "NumMentions": "sum",
                "AvgTone": "mean",
                "EventCode": "count",
            }
        )
        .reset_index()
    )
    # Flatten MultiIndex columns to match ANOMALY_FEATURES order
    weekly.columns = ["week"] + ANOMALY_FEATURES
    weekly = weekly.fillna(0)

    n_weeks = len(weekly)
    if n_weeks < 10:
        print(f"  Warning: {country_code} has only {n_weeks} weeks of data")
    if n_weeks == 0:
        print(f"  Warning: {country_code} has no weekly data, skipping")
        return None, None

    X = weekly[ANOMALY_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200, contamination=0.05, random_state=42
    )
    model.fit(X_scaled)

    models_dir = _models_dir()
    joblib.dump(model, models_dir / f"anomaly_{country_code}.pkl")
    joblib.dump(scaler, models_dir / f"scaler_{country_code}.pkl")
    print(f"  Trained anomaly detector for {country_code} on {n_weeks} weeks")
    return model, scaler


def train_all_anomaly_detectors() -> None:
    """Train Isolation Forest + scaler for all 8 countries. Saves to models/."""
    for country_code in COUNTRIES:
        path = _gdelt_path(country_code)
        if not path.exists():
            print(f"  Warning: no GDELT CSV for {country_code}, skipping")
            continue
        try:
            df = pd.read_csv(path)
            train_anomaly_detector(country_code, df)
        except Exception as e:
            print(f"  Warning: {country_code} failed - {e}")


def detect_anomaly(country_code: str, current_features: dict) -> dict:
    """
    Run anomaly detection on current features.
    Returns dict with anomaly_score (0–1), is_anomaly (bool), severity (LOW/MED/HIGH).
    If model files are missing, returns default LOW.
    """
    models_dir = _models_dir()
    model_path = models_dir / f"anomaly_{country_code}.pkl"
    scaler_path = models_dir / f"scaler_{country_code}.pkl"
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        return {
            "anomaly_score": 0.0,
            "is_anomaly": False,
            "severity": "LOW",
        }

    X = np.array([[current_features.get(f, 0) for f in ANOMALY_FEATURES]])
    X_scaled = scaler.transform(X)
    raw_score = model.score_samples(X_scaled)[0]
    # Normalize to 0–1: score_samples is negative for anomalies; map to [0,1]
    anomaly_score = max(0.0, min(1.0, (-raw_score - 0.3) / 0.7))
    is_anomaly = model.predict(X_scaled)[0] == -1

    severity = (
        "HIGH"
        if anomaly_score > 0.7
        else "MED"
        if anomaly_score > 0.4
        else "LOW"
    )
    return {
        "anomaly_score": round(float(anomaly_score), 3),
        "is_anomaly": bool(is_anomaly),
        "severity": severity,
    }


if __name__ == "__main__":
    print("Training anomaly detectors for all 8 countries...")
    train_all_anomaly_detectors()
    models_dir = _models_dir()
    pkl_files = list(models_dir.glob("*.pkl"))
    n_files = len(pkl_files)
    print(f"Models saved: {n_files} files in {models_dir}")
    if n_files > 0:
        print(f"  Files: {[f.name for f in sorted(pkl_files)]}")

    # Test Ukraine (or first country that has a model)
    print("\nTest detect_anomaly(..., current_features):")
    from backend.ml.data.fetch_gdelt import compute_gdelt_features

    test_country = "UA"
    gdelt_path = _gdelt_path(test_country)
    if gdelt_path.exists():
        df_ua = pd.read_csv(gdelt_path)
        features_30d = compute_gdelt_features(df_ua, window_days=30)
        current = {
            "goldstein_mean": features_30d.get("gdelt_goldstein_mean", 0),
            "goldstein_std": features_30d.get("gdelt_goldstein_std", 0),
            "goldstein_min": features_30d.get("gdelt_goldstein_min", 0),
            "mentions_total": features_30d.get("gdelt_event_count", 0),
            "avg_tone": features_30d.get("gdelt_avg_tone", 0),
            "event_count": features_30d.get("gdelt_event_count", 0),
        }
    else:
        current = {f: 0.0 for f in ANOMALY_FEATURES}
        current["event_count"] = 0
    result = detect_anomaly(test_country, current)
    print(f"  detect_anomaly('{test_country}', current_features) = {result}")
    assert 0 <= result["anomaly_score"] <= 1, "anomaly_score must be in [0, 1]"
    assert result["severity"] in ("LOW", "MED", "HIGH")
    print("  OK: anomaly_score in [0,1], severity in LOW/MED/HIGH.")
