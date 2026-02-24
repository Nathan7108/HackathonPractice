# Sentinel AI — Progress Tracker

## Current State
- [x] S1-01: Project scaffolding
- [x] S1-02: Fetch ACLED data
- [x] S1-03: Fetch GDELT data
- [x] S1-04: Fetch UCDP data
- [x] S1-05: Fetch World Bank data
- [x] S1-06: Feature engineering pipeline
- [x] S2-01: Anomaly detector (Isolation Forest)
- [x] S2-02: Risk scorer (XGBoost)
- [x] S2-03: Sentiment analyzer (FinBERT)
- [x] S2-04: Risk forecaster (LSTM)
- [x] S3-01: FastAPI integration

## Environment
- Python 3.11 (conda env: sentinel)
- Backend: FastAPI on localhost:8000
- Frontend: React+Vite on localhost:5173

## API Keys Required
- ACLED: OAuth (email/password → token)
- OpenAI: OPENAI_API_KEY
- NewsAPI: NEWS_API
- GDELT: None
- UCDP: None
- World Bank: None

## ML Models (status)
- Anomaly Detector: Ready (train via `python -m backend.ml.anomaly` when GDELT data present)
- Risk Scorer (XGBoost): Ready (train via `python -m backend.ml.risk_scorer`; test accuracy ~100%, top 3: acled_fatality_rate, acled_fatalities_30d, acled_geographic_spread; training set 496 country-months)
- Sentiment (FinBERT): PRE-TRAINED (download only)
- Forecaster (LSTM): Ready (train via `python -m backend.ml.forecaster`; saves models/forecaster.pt; forecast_risk() returns 30/60/90 day predictions + trend)

## Monitored Countries
UA, TW, IR, VE, PK, ET, RS, BR

## Fixes (Issue #23 — riskScore vs riskLevel mismatch)
- riskLevel is now always derived from riskScore via thresholds (0–20 LOW, 21–40 MODERATE, 41–60 ELEVATED, 61–80 HIGH, 81–100 CRITICAL).
- riskScore uses weighted probability across all 5 classes (CLASS_WEIGHTS), so score and label never contradict.
- main.py re-derives risk_level after anomaly boost so level stays consistent.
- Branch: feature/fix-risk-score-level-mismatch. PR: create manually with body "Closes #23" if gh not installed.
