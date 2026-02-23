# Sentinel AI — Progress Tracker

## Current State
- [x] S1-01: Project scaffolding
- [x] S1-02: Fetch ACLED data
- [x] S1-03: Fetch GDELT data
- [x] S1-04: Fetch UCDP data
- [ ] S1-05: Fetch World Bank data
- [ ] S1-06: Feature engineering pipeline
- [ ] S2-01: Anomaly detector (Isolation Forest)
- [ ] S2-02: Risk scorer (XGBoost)
- [ ] S2-03: Sentiment analyzer (FinBERT)
- [ ] S2-04: Risk forecaster (LSTM)
- [ ] S3-01: FastAPI integration

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
- Anomaly Detector: NOT TRAINED
- Risk Scorer (XGBoost): NOT TRAINED
- Sentiment (FinBERT): PRE-TRAINED (download only)
- Forecaster (LSTM): NOT TRAINED

## Monitored Countries
UA, TW, IR, VE, PK, ET, RS, BR
