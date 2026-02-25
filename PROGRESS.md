# Sentinel AI — Progress Tracker

## Current State
- [x] S0-01: Next.js scaffold + Tailwind + shadcn/ui + placeholder data (frontend/global-sentinel)
- [x] S0-02: App shell — top nav, tab routing, conditional sidebar
- [x] S0-03: Dashboard tab — KPI cards, Recharts, risk table, regional breakdown
- [x] S0-04: Globe tab — Mapbox 3D globe, risk coloring, detail panel, overlays
- [x] S0-05: Analytics tab — country table, scatter plots, chart grid
- [x] S0-06: Country Intel tab — 11 sections, causal chain, two-column layout
- [x] S0-07: Threat Feed tab — event stream, filters, summary sidebar
- [x] S0-08: Reports tab — intelligence brief library, weekly digest, export placeholders
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
- Frontend: Next.js (global-sentinel) on localhost:3000; legacy React+Vite on localhost:5173

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

## Fixes (Issue #39 — dashboard summary API)
- Added GET `/api/dashboard/summary` to backend/main.py (branch: feature/dashboard-summary-api).
- Endpoint computes all 5 dashboard KPIs from real ML: `SentinelFeaturePipeline.compute_all_countries()` + `predict_risk()` and `detect_anomaly()` per country.
- Response: globalThreatIndex, globalThreatIndexDelta, activeAnomalies, highPlusCountries, highPlusCountriesDelta, escalationAlerts24h, modelHealth (98), countries (sorted by riskScore desc).
- Caching: _dashboard_cache, _dashboard_cache_time, _previous_summary; TTL = CACHE_TTL_SECONDS. predict_risk wrapped in try/except FileNotFoundError like /api/countries.
- Test: `curl http://localhost:8000/api/dashboard/summary`

## Fixes (Issue #40 — dashboard KPI wiring)
- Branch: feature/dashboard-kpi-wiring.
- Added `frontend/global-sentinel/src/lib/api.ts`: API_BASE from NEXT_PUBLIC_API_URL (default http://localhost:8000), `fetchDashboardSummary()`.
- Added `.env.local`: NEXT_PUBLIC_API_URL=http://localhost:8000.
- KpiCardRow: client component with useState(data null, loading true), useEffect calling fetchDashboardSummary(); maps response to Global Threat Index, Active Anomalies, HIGH+ Risk Countries, Escalation Alerts (24h), Model Health; placeholder values when data is null; pulse animation on value text while loading. KpiCard accepts optional `isLoading` for pulse.
- `npm run build` passes.
