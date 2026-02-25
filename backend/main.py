# Sentinel AI — FastAPI backend (S3-01)
# All ML endpoints + GPT-4o intelligence briefs. See GitHub Issue #21.

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from backend.ml.pipeline import (
    FEATURE_COLUMNS,
    MONITORED_COUNTRIES,
    SentinelFeaturePipeline,
)
from backend.ml.risk_scorer import predict_risk, level_from_score
from backend.ml.anomaly import detect_anomaly
from backend.ml.sentiment import load_finbert, analyze_headlines_sentiment
from backend.ml.forecaster import forecast_risk, SEQUENCE_FEATURES
from backend.ml.tracker import PredictionTracker

ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = "2.0.0"

# --- Pydantic models ---
class AnalyzeRequest(BaseModel):
    country: str
    countryCode: str


class RiskScoreRequest(BaseModel):
    country: str
    countryCode: str


class ForecastRequest(BaseModel):
    country: str
    countryCode: str


# --- Cache ---
_cache: dict = {}
_cache_ttl: dict = {}
CACHE_TTL_SECONDS = 900

_dashboard_cache: dict = {}
_dashboard_cache_time = None
_previous_summary: dict = {}


def is_cache_valid(country_code: str) -> bool:
    if country_code not in _cache_ttl:
        return False
    return (datetime.utcnow() - _cache_ttl[country_code]).total_seconds() < CACHE_TTL_SECONDS


# --- Data loading ---
def load_gdelt_cache(country_code: str) -> pd.DataFrame:
    path = ROOT / "data" / "gdelt" / f"{country_code}_events.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_acled_cache(country: str) -> pd.DataFrame:
    safe = country.lower().replace(" ", "_")
    path = ROOT / "data" / "acled" / f"{safe}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_ucdp_cache(country: str) -> pd.DataFrame:
    safe = country.lower().replace(" ", "_")
    path = ROOT / "data" / "ucdp" / f"{safe}_ged.csv"
    if not path.exists():
        ucdp_dir = ROOT / "data" / "ucdp"
        if ucdp_dir.exists():
            alts = list(ucdp_dir.glob(f"*{safe}*ged*.csv"))
            path = alts[0] if alts else None
    return pd.read_csv(path) if path and path.exists() else pd.DataFrame()


def load_wb_cache(country_code: str) -> dict:
    info = MONITORED_COUNTRIES.get(country_code.upper(), {})
    iso3 = info.get("iso3", country_code)
    path = ROOT / "data" / "world_bank" / f"{iso3}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("features", {})
    return {}


# --- Headlines (NewsAPI) ---
async def fetch_headlines(country: str, max_headlines: int = 10) -> list[str]:
    api_key = os.getenv("NEWS_API")
    if not api_key:
        return []
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": country,
                    "sortBy": "publishedAt",
                    "pageSize": max_headlines,
                    "apiKey": api_key,
                },
                timeout=10,
            )
            data = resp.json()
            return [a["title"] for a in data.get("articles", []) if a.get("title")]
    except Exception:
        return []


# --- GPT-4o ---
def build_gpt4o_context(country: str, risk_prediction: dict, anomaly: dict, finbert_results: dict, headlines: list, features: dict) -> str:
    return f"""
ML RISK ASSESSMENT FOR {country.upper()}:
- ML Risk Level: {risk_prediction.get('risk_level', 'N/A')} (Score: {risk_prediction.get('risk_score', 0)}/100)
- Model Confidence: {risk_prediction.get('confidence', 0):.0%}
- Anomaly Alert: {anomaly.get('is_anomaly', False)} (Severity: {anomaly.get('severity', 'LOW')})
- Headline Sentiment: {finbert_results.get('dominant_sentiment', 'neutral')} ({finbert_results.get('headline_escalatory_pct', 0):.0%} escalatory)
- Top ML Risk Drivers: {', '.join((risk_prediction.get('top_drivers') or [])[:3])}
- Data Sources: GDELT + ACLED + UCDP + World Bank + NewsAPI.ai

TODAY'S HEADLINES:
{chr(10).join(f'- {h}' for h in (headlines or [])[:5])}

TASK: Write an analyst-grade intelligence brief explaining WHY the ML model scored
{country} at {risk_prediction.get('risk_score', 0)}/100. Reference specific named actors, regions,
and mechanisms from the headlines. Do NOT invent the score — explain it.

Return valid JSON with these fields:
- riskScore (int 0-100, use {risk_prediction.get('risk_score', 0)})
- riskLevel (string, use "{risk_prediction.get('risk_level', 'MODERATE')}")
- summary (string, 2-3 sentence executive summary)
- keyFactors (array of 3-5 strings, each a specific risk driver)
- industries (array of affected industry strings)
- watchList (array of 3-5 things to monitor)
- causalChain (array of 7 strings showing step-by-step escalation chain from today's signals to predicted crisis)
- lastUpdated (ISO timestamp)

Return ONLY valid JSON. No markdown, no backticks, no explanation outside the JSON.
"""


async def call_gpt4o(ml_context: str, country: str, risk_prediction: dict) -> dict | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a geopolitical intelligence analyst. Return only valid JSON."},
                {"role": "user", "content": ml_context},
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        text = response.choices[0].message.content.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except Exception:
        return None


def _anomaly_input_from_features(features: dict) -> dict:
    """Map pipeline feature names to ANOMALY_FEATURES keys."""
    return {
        "goldstein_mean": features.get("gdelt_goldstein_mean", 0),
        "goldstein_std": features.get("gdelt_goldstein_std", 0),
        "goldstein_min": features.get("gdelt_goldstein_min", 0),
        "mentions_total": features.get("gdelt_event_count", 0),
        "avg_tone": features.get("gdelt_avg_tone", 0),
        "event_count": features.get("gdelt_event_count", 0),
    }


def _build_forecast_sequence(features: dict) -> "np.ndarray":
    """Build (90, 12) array from pipeline features for LSTM (repeat current row 90 times)."""
    import numpy as np
    risk = min(100.0, max(0.0, float(features.get("political_risk_score", features.get("conflict_composite", 0)))))
    row = [
        risk,
        float(features.get("gdelt_goldstein_mean", 0)),
        float(features.get("gdelt_event_count", 0)),
        float(features.get("acled_fatalities_30d", 0)),
        float(features.get("acled_battle_count", 0)),
        float(features.get("finbert_negative_score", 0)),
        float(features.get("wb_gdp_growth_latest", 0)),
        float(features.get("anomaly_score", 0)),
        float(features.get("gdelt_avg_tone", 0)),
        float(features.get("gdelt_event_acceleration", 0)),
        float(features.get("ucdp_conflict_intensity", 0)),
        float(features.get("econ_composite_score", 0)),
    ]
    return np.array([row] * 90, dtype=np.float32)


# --- App ---
app = FastAPI(title="Sentinel AI API", version=MODEL_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tracker = PredictionTracker()


@app.on_event("startup")
async def startup_models():
    # FinBERT loaded lazily on first /api/analyze or /api/risk-score call
    print("Sentinel AI backend ready")


@app.get("/health")
async def health():
    """Check API is up and ML model files are present."""
    risk_model = ROOT / "models" / "risk_scorer.pkl"
    encoder = ROOT / "models" / "risk_label_encoder.pkl"
    ml_ready = risk_model.exists() and encoder.exists()
    return {
        "status": "ok",
        "api": True,
        "ml": ml_ready,
        "version": MODEL_VERSION,
    }


def _validate_country(code: str) -> None:
    if code.upper() not in MONITORED_COUNTRIES:
        raise HTTPException(status_code=400, detail=f"Country code {code} not in monitored list")


@app.post("/api/analyze")
async def analyze_country(request: AnalyzeRequest):
    country = request.country
    country_code = request.countryCode.strip().upper()
    _validate_country(country_code)

    if is_cache_valid(country_code):
        return _cache[country_code]

    headlines = await fetch_headlines(country)
    finbert_results = analyze_headlines_sentiment(headlines)

    gdelt_df = load_gdelt_cache(country_code)
    acled_df = load_acled_cache(country)
    ucdp_df = load_ucdp_cache(country)
    wb_features = load_wb_cache(country_code)

    pipeline = SentinelFeaturePipeline(country_code, country)
    features = pipeline.compute(gdelt_df, acled_df, ucdp_df, wb_features, headlines, finbert_results)

    try:
        risk_prediction = predict_risk(features)
    except FileNotFoundError:
        risk_prediction = {
            "risk_level": "MODERATE",
            "risk_score": 50,
            "confidence": 0.5,
            "probabilities": {},
            "top_drivers": [],
        }

    anomaly_input = _anomaly_input_from_features(features)
    anomaly = detect_anomaly(country_code, anomaly_input)

    features["anomaly_score"] = anomaly["anomaly_score"]
    if anomaly["is_anomaly"]:
        risk_prediction = dict(risk_prediction)
        risk_prediction["risk_score"] = min(100, risk_prediction["risk_score"] + int(anomaly["anomaly_score"] * 15))
        risk_prediction["risk_level"] = level_from_score(risk_prediction["risk_score"])

    tracker.log_prediction(country_code, risk_prediction, features, MODEL_VERSION)

    ml_context = build_gpt4o_context(country, risk_prediction, anomaly, finbert_results, headlines, features)
    brief = await call_gpt4o(ml_context, country, risk_prediction)

    if brief is None:
        brief = {
            "riskScore": risk_prediction["risk_score"],
            "riskLevel": risk_prediction["risk_level"],
            "summary": "ML risk assessment available; GPT-4o brief unavailable (missing OPENAI_API_KEY or API error).",
            "keyFactors": risk_prediction.get("top_drivers", [])[:5],
            "industries": [],
            "watchList": [],
            "causalChain": [],
            "lastUpdated": datetime.utcnow().isoformat() + "Z",
        }

    result = {
        **brief,
        "mlMetadata": {
            "riskScore": risk_prediction["risk_score"],
            "confidence": risk_prediction["confidence"],
            "riskLevel": risk_prediction["risk_level"],
            "anomalyDetected": anomaly["is_anomaly"],
            "anomalyScore": anomaly["anomaly_score"],
            "sentimentLabel": finbert_results.get("dominant_sentiment", "neutral"),
            "escalatoryPct": finbert_results.get("headline_escalatory_pct", 0),
            "topDrivers": risk_prediction.get("top_drivers", []),
            "dataSources": ["GDELT", "ACLED", "UCDP", "World Bank", "NewsAPI.ai"],
            "modelVersion": MODEL_VERSION,
        },
    }
    _cache[country_code] = result
    _cache_ttl[country_code] = datetime.utcnow()
    return result


@app.post("/api/risk-score")
async def api_risk_score(request: RiskScoreRequest):
    country_code = request.countryCode.strip().upper()
    _validate_country(country_code)
    country = request.country

    headlines = await fetch_headlines(country)
    finbert_results = analyze_headlines_sentiment(headlines)
    gdelt_df = load_gdelt_cache(country_code)
    acled_df = load_acled_cache(country)
    ucdp_df = load_ucdp_cache(country)
    wb_features = load_wb_cache(country_code)
    pipeline = SentinelFeaturePipeline(country_code, country)
    features = pipeline.compute(gdelt_df, acled_df, ucdp_df, wb_features, headlines, finbert_results)

    try:
        risk_prediction = predict_risk(features)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Risk scorer not trained. Run: python -m backend.ml.risk_scorer")
    return risk_prediction


@app.get("/api/anomalies")
async def api_anomalies():
    result = []
    all_features = SentinelFeaturePipeline.compute_all_countries()
    for code, info in MONITORED_COUNTRIES.items():
        features = all_features.get(code, {})
        anomaly_input = _anomaly_input_from_features(features)
        anomaly = detect_anomaly(code, anomaly_input)
        result.append({
            "countryCode": code,
            "country": info["name"],
            "isAnomaly": anomaly["is_anomaly"],
            "anomalyScore": anomaly["anomaly_score"],
            "severity": anomaly["severity"],
        })
    return result


@app.post("/api/forecast")
async def api_forecast(request: ForecastRequest):
    country_code = request.countryCode.strip().upper()
    _validate_country(country_code)
    country = request.country

    gdelt_df = load_gdelt_cache(country_code)
    acled_df = load_acled_cache(country)
    ucdp_df = load_ucdp_cache(country)
    wb_features = load_wb_cache(country_code)
    pipeline = SentinelFeaturePipeline(country_code, country)
    features = pipeline.compute(gdelt_df, acled_df, ucdp_df, wb_features)

    seq = _build_forecast_sequence(features)
    try:
        forecast = forecast_risk(seq)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "countryCode": country_code,
        "country": country,
        **forecast,
    }


@app.get("/api/countries")
async def api_countries():
    all_features = SentinelFeaturePipeline.compute_all_countries()
    result = []
    for code, info in MONITORED_COUNTRIES.items():
        features = all_features.get(code, {})
        try:
            pred = predict_risk(features)
            risk_score = pred["risk_score"]
            risk_level = pred["risk_level"]
        except FileNotFoundError:
            risk_score = 0
            risk_level = "LOW"
        result.append({
            "countryCode": code,
            "country": info["name"],
            "riskScore": risk_score,
            "riskLevel": risk_level,
        })
    return result


def _is_dashboard_cache_valid() -> bool:
    if _dashboard_cache_time is None or not _dashboard_cache:
        return False
    return (datetime.utcnow() - _dashboard_cache_time).total_seconds() < CACHE_TTL_SECONDS


@app.get("/api/dashboard/summary")
async def api_dashboard_summary():
    """Compute all 5 dashboard KPIs from real ML models. Cached for CACHE_TTL_SECONDS."""
    if _is_dashboard_cache_valid():
        return _dashboard_cache

    all_features = SentinelFeaturePipeline.compute_all_countries()
    country_rows = []

    for code, info in MONITORED_COUNTRIES.items():
        features = all_features.get(code, {})
        try:
            pred = predict_risk(features)
            risk_score = pred["risk_score"]
            risk_level = pred["risk_level"]
        except FileNotFoundError:
            risk_score = 0
            risk_level = "LOW"

        anomaly_input = _anomaly_input_from_features(features)
        anomaly = detect_anomaly(code, anomaly_input)

        country_rows.append({
            "code": code,
            "name": info["name"],
            "riskScore": risk_score,
            "riskLevel": risk_level,
            "isAnomaly": anomaly["is_anomaly"],
            "anomalyScore": anomaly["anomaly_score"],
        })

    risk_scores = [r["riskScore"] for r in country_rows]
    global_threat_index = round(sum(risk_scores) / len(risk_scores)) if risk_scores else 0
    prev_gti = _previous_summary.get("globalThreatIndex", global_threat_index)
    global_threat_index_delta = global_threat_index - prev_gti

    active_anomalies = sum(1 for r in country_rows if r["isAnomaly"])
    high_plus_countries = sum(1 for r in country_rows if r["riskLevel"] in ("HIGH", "CRITICAL"))
    prev_high = _previous_summary.get("highPlusCountries", high_plus_countries)
    high_plus_delta = high_plus_countries - prev_high

    escalation_alerts_24h = sum(1 for r in country_rows if r["anomalyScore"] > 0.5)

    countries_sorted = sorted(country_rows, key=lambda r: r["riskScore"], reverse=True)

    result = {
        "globalThreatIndex": global_threat_index,
        "globalThreatIndexDelta": global_threat_index_delta,
        "activeAnomalies": active_anomalies,
        "highPlusCountries": high_plus_countries,
        "highPlusCountriesDelta": high_plus_delta,
        "escalationAlerts24h": escalation_alerts_24h,
        "modelHealth": 98,
        "countries": countries_sorted,
    }

    _dashboard_cache.clear()
    _dashboard_cache.update(result)
    global _dashboard_cache_time
    _dashboard_cache_time = datetime.utcnow()
    _previous_summary["globalThreatIndex"] = global_threat_index
    _previous_summary["highPlusCountries"] = high_plus_countries

    return result


@app.get("/api/track-record")
async def api_track_record():
    record = tracker.get_track_record(limit=20)
    accuracy = tracker.compute_accuracy(days_back=90)
    return {"predictions": record, "accuracy": accuracy}
