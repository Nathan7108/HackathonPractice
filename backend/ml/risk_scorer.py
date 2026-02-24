# Sentinel AI — XGBoost risk scorer (S2-02)
# 47 features -> 5 risk levels (LOW/MODERATE/ELEVATED/HIGH/CRITICAL) with confidence.
# See GitHub Issue #16.

import warnings
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from backend.ml.pipeline import FEATURE_COLUMNS, MONITORED_COUNTRIES

RISK_LABELS = ["LOW", "MODERATE", "ELEVATED", "HIGH", "CRITICAL"]

# Weights for weighted probability score — higher classes contribute more (0-100).
CLASS_WEIGHTS = {
    "LOW": 0,
    "MODERATE": 25,
    "ELEVATED": 50,
    "HIGH": 75,
    "CRITICAL": 100,
}

from sklearn.preprocessing import LabelEncoder

from backend.ml.pipeline import FEATURE_COLUMNS, MONITORED_COUNTRIES

RISK_LABELS = ["LOW", "MODERATE", "ELEVATED", "HIGH", "CRITICAL"]


def score_from_probabilities(probabilities: list, labels: list) -> int:
    """
    Compute a 0-100 risk score as the weighted average of class probabilities.
    Ensures score is semantically aligned with the class distribution.
    """
    score = 0.0
    for prob, label in zip(probabilities, labels):
        score += prob * CLASS_WEIGHTS.get(str(label), 0)
    return max(0, min(100, int(round(score))))


def level_from_score(score: int) -> str:
    """Deterministic mapping — score and level ALWAYS agree. Thresholds: 0-20 LOW, 21-40 MODERATE, 41-60 ELEVATED, 61-80 HIGH, 81-100 CRITICAL."""
    if score >= 81:
        return "CRITICAL"
    if score >= 61:
        return "HIGH"
    if score >= 41:
        return "ELEVATED"
    if score >= 21:
        return "MODERATE"
    return "LOW"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _acled_features_from_group(group: pd.DataFrame, window_days: int = 30) -> dict:
    """
    Compute 10 ACLED features from a single group (e.g. one month of events).
    Used for training data where we don't have 'now' relative windows.
    """
    if group is None or group.empty:
        return {
            "acled_fatalities_30d": 0.0,
            "acled_battle_count": 0,
            "acled_civilian_violence": 0,
            "acled_explosion_count": 0,
            "acled_protest_count": 0,
            "acled_fatality_rate": 0.0,
            "acled_event_count_90d": 0,
            "acled_event_acceleration": 0.0,
            "acled_unique_actors": 0,
            "acled_geographic_spread": 0,
        }
    fatalities = pd.to_numeric(group.get("fatalities", pd.Series(dtype=float)), errors="coerce").fillna(0)
    total_fatal = float(fatalities.sum())
    n = len(group)
    event_type = group.get("event_type", pd.Series(dtype=object))
    event_type = event_type.astype(str)
    return {
        "acled_fatalities_30d": total_fatal,
        "acled_battle_count": int((event_type == "Battles").sum()),
        "acled_civilian_violence": int((event_type == "Violence against civilians").sum()),
        "acled_explosion_count": int((event_type == "Explosions/Remote violence").sum()),
        "acled_protest_count": int(event_type.isin(["Protests", "Riots"]).sum()),
        "acled_fatality_rate": total_fatal / max(window_days, 1),
        "acled_event_count_90d": n,
        "acled_event_acceleration": 1.0,
        "acled_unique_actors": int(group["actor1"].nunique()) if "actor1" in group.columns else 0,
        "acled_geographic_spread": int(group["admin1"].nunique()) if "admin1" in group.columns else 0,
    }


def _label_from_fatalities(fatalities: float) -> str:
    if fatalities >= 500:
        return "CRITICAL"
    if fatalities >= 200:
        return "HIGH"
    if fatalities >= 50:
        return "ELEVATED"
    if fatalities >= 10:
        return "MODERATE"
    return "LOW"


def build_training_dataset() -> pd.DataFrame:
    """
    Build labeled training dataset from ACLED (country-month aggregation).
    Returns DataFrame with FEATURE_COLUMNS + 'risk_label' + 'country_code'.
    """
    root = _repo_root()
    data_acled = root / "data" / "acled"
    frames = []

    for code, info in MONITORED_COUNTRIES.items():
        acled_name = info["acled_name"]
        safe = acled_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("__", "_")
        ged_candidates = list(data_ucdp.glob(f"*{safe.split('_')[0]}*ged*.csv")) + list(data_ucdp.glob(f"*{acled_name.split()[0].lower()}*ged*.csv"))
        ged_path = data_ucdp / f"{safe}_ged.csv"
        if not ged_path.exists() and ged_candidates:
            ged_path = Path(ged_candidates[0])
        if ged_path.exists():
            try:
                ged_df = pd.read_csv(ged_path)
                ucdp_by_country[code] = compute_ucdp_features(ged_df, window_years=5)
                ucdp_ged_paths[code] = ged_path
            except Exception as e:
                warnings.warn(f"UCDP {code}: {e}")
                ucdp_by_country[code] = {k: 0 for k in ["ucdp_total_deaths", "ucdp_state_conflict_years", "ucdp_civilian_deaths", "ucdp_conflict_intensity", "ucdp_recurrence_rate"]}
        else:
            ucdp_by_country[code] = {k: 0 for k in ["ucdp_total_deaths", "ucdp_state_conflict_years", "ucdp_civilian_deaths", "ucdp_conflict_intensity", "ucdp_recurrence_rate"]}
        wb_path = data_wb / f"{iso3}.json"
        if wb_path.exists():
            try:
                with open(wb_path, encoding="utf-8") as f:
                    wb_by_country[code] = json.load(f).get("features", {})
            except Exception as e:
                warnings.warn(f"World Bank {code}: {e}")
                wb_by_country[code] = {}
        else:
            wb_by_country[code] = {}

    wb_feature_keys = [k for k in FEATURE_COLUMNS if k.startswith("wb_") or k == "econ_composite_score"]

    # Country-month rows from ACLED + GDELT + UCDP + WB + sentiment + derived
    for code, info in MONITORED_COUNTRIES.items():
        acled_name = info["acled_name"]
        iso3 = info["iso3"]
    Build labeled training dataset from ACLED (country-month aggregation).
    Returns DataFrame with FEATURE_COLUMNS + 'risk_label' + 'country_code'.
    """
    root = _repo_root()
    data_acled = root / "data" / "acled"
    frames = []

    for code, info in MONITORED_COUNTRIES.items():
        acled_name = info["acled_name"]
        acled_file = acled_name.lower().replace(" ", "_") + ".csv"
        acled_path = data_acled / acled_file
        if not acled_path.exists():
            continue
        try:
            acled_df = pd.read_csv(acled_path)
        except Exception as e:
            warnings.warn(f"ACLED {code} read failed: {e}")
            continue
        if "event_date" not in acled_df.columns or acled_df.empty:
            continue
        acled_df["event_date"] = pd.to_datetime(acled_df["event_date"], errors="coerce")
        acled_df = acled_df.dropna(subset=["event_date"])
        if acled_df.empty:
            continue
        acled_df["year_month"] = acled_df["event_date"].dt.to_period("M")

        for period, group in acled_df.groupby("year_month"):
            features = _acled_features_from_group(group, window_days=30)
            for col in FEATURE_COLUMNS:
                if col not in f or f[col] is None:
                    f[col] = 0.0
            fatalities = float(pd.to_numeric(group.get("fatalities", pd.Series(0)), errors="coerce").fillna(0).sum())
            f["risk_label"] = _label_from_fatalities(fatalities)
            f["country_code"] = code
            frames.append(f)

    # UCDP historical training rows
    ac_paths = list(data_ucdp.glob("armed_conflict*.csv")) + list(data_ucdp.glob("UcdpPrio*.csv")) + list(data_ucdp.glob("*acd*.csv"))
    ac_df = pd.DataFrame()
    for p in ac_paths:
        try:
            ac_df = pd.read_csv(p)
            if "year" in ac_df.columns and "intensity_level" in ac_df.columns or "intensity" in ac_df.columns:
                break
        except Exception:
            continue
    if ac_df.empty and (data_ucdp / "armed_conflict_1946_2023.csv").exists():
        ac_df = pd.read_csv(data_ucdp / "armed_conflict_1946_2023.csv")
    if not ac_df.empty:
        ucdp_labels_df = build_ucdp_training_labels(ac_df)
        gwno_col = "gwno_loc" if "gwno_loc" in ac_df.columns else "gwnoloc" if "gwnoloc" in ac_df.columns else None
        if gwno_col:
            merge_df = ac_df[["location", "year", gwno_col]].drop_duplicates(subset=["location", "year"], keep="first")
            ucdp_labels_df = ucdp_labels_df.merge(merge_df, on=["location", "year"], how="left")
        loc_to_iso2 = {"Ukraine": "UA", "Taiwan": "TW", "Iran (Government of)": "IR", "Iran": "IR", "Venezuela": "VE", "Pakistan": "PK", "Ethiopia": "ET", "Serbia": "RS", "Brazil": "BR"}
        for _, row in ucdp_labels_df.iterrows():
            gwno_val = row.get("gwno_loc", row.get("gwnoloc", None))
            country_code = None
            if gwno_val is not None and str(gwno_val).strip() and str(gwno_val) != "nan":
                try:
                    gwno_int = int(float(str(gwno_val).split(",")[0].strip()))
                    country_code = GW_TO_ISO2.get(gwno_int)
                except (ValueError, TypeError):
                    pass
            if country_code is None:
                loc = str(row.get("location", ""))
                country_code = loc_to_iso2.get(loc) or (loc_to_iso2.get(loc.split(",")[0].strip()) if "," in loc else None)
            if country_code not in MONITORED_COUNTRIES:
                continue
            ucdp_feat = ucdp_by_country.get(country_code, {k: 0 for k in ["ucdp_total_deaths", "ucdp_state_conflict_years", "ucdp_civilian_deaths", "ucdp_conflict_intensity", "ucdp_recurrence_rate"]})
            wb_feat = {k: wb_by_country.get(country_code, {}).get(k, 0.0) for k in wb_feature_keys}
            wb_raw = wb_by_country.get(country_code, {})
            if wb_raw:
                from backend.ml.data.fetch_world_bank import format_wb_features
                wb_feat = {k: format_wb_features(wb_raw).get(k, 0.0) for k in wb_feature_keys}
            else:
                wb_feat = {k: 0.0 for k in wb_feature_keys}
            f = {col: 0.0 for col in FEATURE_COLUMNS}
            f.update(ucdp_feat)
            f.update(wb_feat)
            f.update(_sentiment_random(rng))
            f.update(_derived_features_train(f))
            for col in FEATURE_COLUMNS:
                if col not in f or f[col] is None:
                    f[col] = 0.0
            f["risk_label"] = str(row.get("risk_label", "LOW"))
            f["country_code"] = country_code
            frames.append(f)
        for period, group in acled_df.groupby("year_month"):
            features = _acled_features_from_group(group, window_days=30)
            for col in FEATURE_COLUMNS:
                if col not in features:
                    features[col] = 0.0
            fatalities = float(
                pd.to_numeric(group.get("fatalities", pd.Series(0)), errors="coerce").fillna(0).sum()
            )
            features["risk_label"] = _label_from_fatalities(fatalities)
            features["country_code"] = code
            frames.append(features)

    if not frames:
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["risk_label", "country_code"])

    df = pd.DataFrame(frames)
    int_cols = {
        "gdelt_event_count", "acled_battle_count", "acled_civilian_violence", "acled_explosion_count",
        "acled_protest_count", "acled_event_count_90d", "acled_unique_actors", "acled_geographic_spread",
        "ucdp_state_conflict_years", "headline_volume",
        "gdelt_event_count",
        "acled_battle_count",
        "acled_civilian_violence",
        "acled_explosion_count",
        "acled_protest_count",
        "acled_event_count_90d",
        "acled_unique_actors",
        "acled_geographic_spread",
        "ucdp_state_conflict_years",
        "headline_volume",
    }
    for c in int_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df


def train_risk_scorer(training_df: pd.DataFrame):
    """
    Train XGBoost classifier on FEATURE_COLUMNS, save model and label encoder.
    Prints classification_report and top 10 feature importances.
    """
    root = _repo_root()
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if training_df.empty or "risk_label" not in training_df.columns:
        raise ValueError("training_df must have risk_label and at least one row")

    # Merge CRITICAL into HIGH if CRITICAL has too few samples for stratify
    df = training_df.copy()
    label_counts = df["risk_label"].value_counts()
    if "CRITICAL" in label_counts and label_counts.get("CRITICAL", 0) < 2:
        df.loc[df["risk_label"] == "CRITICAL", "risk_label"] = "HIGH"
    if "HIGH" in label_counts and label_counts.get("HIGH", 0) < 2:
        df.loc[df["risk_label"] == "HIGH", "risk_label"] = "ELEVATED"

    le = LabelEncoder()
    le.fit(RISK_LABELS)
    y = le.transform(df["risk_label"].astype(str))
    X = df[FEATURE_COLUMNS].fillna(0).astype(float)
    X = df[FEATURE_COLUMNS].fillna(0)

    stratify_arg = y if all((y == i).sum() >= 2 for i in range(len(le.classes_))) else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_arg
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Normalize all 47 features to 0-1 so World Bank (large magnitude) doesn't dominate
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Upweight HIGH and CRITICAL in loss to improve their recall
    high_idx = le.transform(["HIGH"])[0]
    crit_idx = le.transform(["CRITICAL"])[0]
    sample_weight = np.where(
        np.isin(y_train, [high_idx, crit_idx]), 2.0, 1.0
    )

    fit_kw = {"sample_weight": sample_weight}
    fit_kw = {}
    if len(X_test) >= 10:
        fit_kw["eval_set"] = [(X_test, y_test)]
        fit_kw["verbose"] = 50

    model_kw = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
    model_kw = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
    )
    if len(X_test) >= 10:
        model_kw["early_stopping_rounds"] = 20
    model = xgb.XGBClassifier(**model_kw)
    model.fit(X_train, y_train, **fit_kw)

    y_pred = model.predict(X_test)
    target_names = [le.inverse_transform([i])[0] for i in range(len(le.classes_))]
    print("Classification report (test set):")
    print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

    importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print("\nTop 10 Feature Importances:")
    print(importance.nlargest(10))

    model_path = models_dir / "risk_scorer.pkl"
    encoder_path = models_dir / "risk_label_encoder.pkl"
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nSaved: {model_path}, {encoder_path}, {scaler_path}")
    joblib.dump(model, model_path)
    joblib.dump(le, encoder_path)
    print(f"\nSaved: {model_path}, {encoder_path}")
    return model


def predict_risk(features: dict) -> dict:
    """
    Load trained model and predict risk from a 47-feature dict (e.g. from SentinelFeaturePipeline.compute()).
    Returns dict with risk_level, risk_score (0-100), confidence, probabilities, top_drivers (5 names).
    risk_level is always derived from risk_score thresholds so they never contradict.
    Applies the same MinMaxScaler used at training. Returns dict with risk_level, risk_score (0-100),
    confidence, probabilities, top_drivers (5 names).
    Returns dict with risk_level, risk_score (0-100), confidence, probabilities, top_drivers (5 names).
    """
    root = _repo_root()
    model_path = root / "models" / "risk_scorer.pkl"
    encoder_path = root / "models" / "risk_label_encoder.pkl"
    if not model_path.exists() or not encoder_path.exists():
        raise FileNotFoundError("Train the risk scorer first: python -m backend.ml.risk_scorer")

    model = joblib.load(model_path)
    le = joblib.load(encoder_path)

    X = pd.DataFrame([{col: features.get(col, 0) for col in FEATURE_COLUMNS}]).astype(float)
    X = scaler.transform(X)
    if not model_path.exists() or not encoder_path.exists():
        raise FileNotFoundError("Train the risk scorer first: python -m backend.ml.risk_scorer")

    model = joblib.load(model_path)
    le = joblib.load(encoder_path)

    X = pd.DataFrame([{col: features.get(col, 0) for col in FEATURE_COLUMNS}])
    probabilities = model.predict_proba(X)[0]
    # Ordered labels matching probabilities array (model/encoder order)
    labels = list(le.inverse_transform(range(len(probabilities))))

    # Weighted score from all class probabilities — semantically aligned with distribution
    risk_score = score_from_probabilities(probabilities.tolist(), labels)
    # Level derived from score — ALWAYS consistent
    risk_level = level_from_score(risk_score)

    # Confidence = probability of the derived level's class
    level_idx = labels.index(risk_level) if risk_level in labels else 0
    confidence = float(probabilities[level_idx])

    importance = model.feature_importances_
    top_features = sorted(
        zip(FEATURE_COLUMNS, importance), key=lambda x: -x[1]
    )[:5]
    top_drivers = [str(f) for f, _ in top_features]

    proba_dict = {
        str(labels[i]): round(float(probabilities[i]), 3)
        for i in range(len(probabilities))
    }
    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "confidence": round(confidence, 3),
        "probabilities": proba_dict,
        "top_drivers": top_drivers,
    }


if __name__ == "__main__":
    print("Building training dataset from ACLED...")
    training_df = build_training_dataset()
    print(f"Total training rows: {len(training_df)}")
    print(f"Dataset shape: {training_df.shape} (47 features + risk_label + country_code)")
    print("Building training dataset from ACLED...")
    training_df = build_training_dataset()
    print(f"Dataset shape: {training_df.shape}")
    if training_df.empty:
        print("No training data. Ensure data/acled/*.csv exist for monitored countries.")
        raise SystemExit(1)
    print("Risk label distribution:")
    print(training_df["risk_label"].value_counts())

    print("\nTraining XGBoost risk scorer...")
    train_risk_scorer(training_df)

    print("\nTesting predict_risk() with sample features from pipeline...")
    from backend.ml.pipeline import SentinelFeaturePipeline

    pipeline = SentinelFeaturePipeline("UA", "Ukraine")
    try:
        all_features = SentinelFeaturePipeline.compute_all_countries()
        ua_features = all_features.get("UA")
        if ua_features is not None:
            ua_feat = {k: ua_features.get(k, 0) for k in FC}
            ukraine_result = predict_risk(ua_feat)
            print("Ukraine prediction:", ukraine_result)
            if ukraine_result["risk_level"] not in ("HIGH", "CRITICAL"):
                print("  [Note: Ukraine expected HIGH or CRITICAL; check feature flow from pipeline to scorer.]")
        else:
            sample = next(iter(all_features.values())) if all_features else {k: 0.0 for k in FC}
            print("Ukraine not in results; sample prediction:", predict_risk({k: sample.get(k, 0) for k in FC}))
    except Exception as e:
        print("Pipeline/predict_risk error:", e)
    print("\nTesting predict_risk() with sample features from pipeline...")
    from backend.ml.pipeline import SentinelFeaturePipeline

    pipeline = SentinelFeaturePipeline("UA", "Ukraine")
    try:
        all_features = SentinelFeaturePipeline.compute_all_countries()
        sample = all_features.get("UA") or next(iter(all_features.values()))
    except Exception:
        sample = {k: 0.0 for k in FEATURE_COLUMNS}
    result = predict_risk(sample)
    print("Sample prediction:", result)
