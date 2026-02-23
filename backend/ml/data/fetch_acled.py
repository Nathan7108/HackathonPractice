# Sentinel AI — fetch ACLED conflict data (S1-02)
# New ACLED API: OAuth at acleddata.com, Bearer token, pagination 5000/page.

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ISO2 -> full country name (ACLED uses full names)
COUNTRIES = {
    "UA": "Ukraine",
    "TW": "Taiwan",
    "IR": "Iran",
    "VE": "Venezuela",
    "PK": "Pakistan",
    "ET": "Ethiopia",
    "RS": "Serbia",
    "BR": "Brazil",
}

TOKEN_URL = "https://acleddata.com/oauth/token"
LOGIN_URL = "https://acleddata.com/user/login?_format=json"
API_URL = "https://acleddata.com/api/acled/read?_format=json"
PAGE_LIMIT = 5000
FIELDS = "event_date|event_type|sub_event_type|fatalities|actor1|actor2|admin1|latitude|longitude|notes"


def get_acled_token(username: str, password: str) -> str:
    """Get OAuth Bearer token from ACLED. New API (2025+)."""
    resp = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "username": username,
            "password": password,
            "grant_type": "password",
            "client_id": "acled",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def get_acled_session(username: str, password: str) -> requests.Session:
    """
    Cookie-based login (same as browser). Use when OAuth returns 403.
    Docs: https://acleddata.com/api-documentation/getting-started
    """
    session = requests.Session()
    session.headers["Content-Type"] = "application/json"
    resp = session.post(
        LOGIN_URL,
        json={"name": username, "pass": password},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "current_user" not in data:
        raise RuntimeError("Cookie login failed: no current_user in response")
    return session


def fetch_acled(
    country: str,
    token: str | None = None,
    session: requests.Session | None = None,
    start_date: str = "2020-01-01",
    end_date: str = "2026-02-28",
) -> pd.DataFrame:
    """
    Fetch ACLED events for one country with pagination.
    Use either token (OAuth) or session (cookie-based). If both set, token is used.
    Returns DataFrame with columns: event_date, event_type, sub_event_type,
    fatalities, actor1, actor2, admin1, latitude, longitude, notes.
    """
    if token:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        get = requests.get
        req_kw = {"headers": headers}
    elif session:
        headers = {"Content-Type": "application/json"}
        get = session.get
        req_kw = {"headers": headers}
    else:
        raise ValueError("Provide either token or session")

    all_data = []
    page = 1

    while True:
        params = {
            "country": country,
            "event_date": f"{start_date}|{end_date}",
            "event_date_where": "BETWEEN",
            "fields": FIELDS,
            "limit": PAGE_LIMIT,
            "page": page,
        }
        try:
            response = get(
                API_URL,
                params=params,
                timeout=60,
                **req_kw,
            )
            if response.status_code == 403:
                try:
                    body = response.json()
                    msg = body.get("message", body.get("error", response.text or "Access denied"))
                except Exception:
                    msg = response.text or "Access denied"
                print(
                    f"  Warning: 403 Forbidden for {country}. "
                    f"ACLED says: {msg}. "
                    "Log in at https://acleddata.com: accept consent, complete profile, ensure your account has API access."
                )
                break
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"  Warning: fetch failed for {country} page {page}: {e}")
            break

        rows = data.get("data", [])
        if not rows:
            break

        # API may return list of dicts or list of pipe-separated strings
        if rows and isinstance(rows[0], str):
            # Parse pipe-separated: event_date|event_type|...
            keys = [k.strip() for k in FIELDS.split("|")]
            all_data.extend(
                dict(zip(keys, (c.strip() for c in r.split("|")))) for r in rows
            )
        else:
            all_data.extend(rows)

        if len(rows) < PAGE_LIMIT:
            break
        page += 1

    return pd.DataFrame(all_data) if all_data else pd.DataFrame()


def compute_acled_features(df: pd.DataFrame, window_days: int = 30) -> dict:
    """
    Compute 10 ML-ready ACLED features per ML Guide Section 2.2.
    Returns dict with exactly: acled_fatalities_30d, acled_battle_count,
    acled_civilian_violence, acled_explosion_count, acled_protest_count,
    acled_fatality_rate, acled_event_count_90d, acled_event_acceleration,
    acled_unique_actors, acled_geographic_spread.
    """
    if df is None or df.empty:
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

    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"])
    now = pd.Timestamp.now()
    recent = df[df["event_date"] > now - pd.Timedelta(days=window_days)]
    recent_90 = df[df["event_date"] > now - pd.Timedelta(days=90)]

    fatalities = pd.to_numeric(recent["fatalities"], errors="coerce").fillna(0)
    count_90 = len(recent_90)
    count_30 = len(recent)
    older_60_90 = max(count_90 - count_30, 1)

    return {
        "acled_fatalities_30d": float(fatalities.sum()),
        "acled_battle_count": int(len(recent[recent["event_type"] == "Battles"])),
        "acled_civilian_violence": int(
            len(recent[recent["event_type"] == "Violence against civilians"])
        ),
        "acled_explosion_count": int(
            len(recent[recent["event_type"] == "Explosions/Remote violence"])
        ),
        "acled_protest_count": int(
            len(
                recent[
                    recent["event_type"].isin(["Protests", "Riots"])
                ]
            )
        ),
        "acled_fatality_rate": float(fatalities.sum() / max(window_days, 1)),
        "acled_event_count_90d": count_90,
        "acled_event_acceleration": float(count_30 / older_60_90),
        "acled_unique_actors": int(recent["actor1"].nunique()) if "actor1" in recent.columns else 0,
        "acled_geographic_spread": int(recent["admin1"].nunique()) if "admin1" in recent.columns else 0,
    }


def _repo_root() -> Path:
    """Project root (where .env lives). backend/ml/data/fetch_acled.py -> repo root."""
    return Path(__file__).resolve().parents[3]


def _data_dir() -> Path:
    """Project data dir: data/acled/ (from repo root)."""
    d = _repo_root() / "data" / "acled"
    d.mkdir(parents=True, exist_ok=True)
    return d


if __name__ == "__main__":
    # Load .env from project root so it works regardless of cwd
    env_path = _repo_root() / ".env"
    load_dotenv(dotenv_path=env_path)
    if not env_path.exists():
        print(f"Note: No .env at {env_path}")
    username = os.getenv("ACLED_USERNAME")
    password = os.getenv("ACLED_PASSWORD")
    token = os.getenv("ACLED_ACCESS_TOKEN", "").strip()

    session = None
    # Prefer OAuth token first (so new token gets used)
    if token:
        print("Using ACLED_ACCESS_TOKEN from .env.")
        r = requests.get(
            API_URL,
            params={"country": "Ukraine", "limit": 1},
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            timeout=30,
        )
        if r.status_code == 200:
            print("Token valid. Fetching data for 8 countries (first request may take 30–60s)...")
        elif r.status_code == 403 and username and password:
            print("Token returned 403. Trying cookie-based login (user/login)...")
            try:
                session = get_acled_session(username, password)
                print("Session started. Using cookie auth for all requests.")
                token = None
            except Exception as e:
                print(f"Cookie login failed: {e}")
        else:
            print(f"Token test got HTTP {r.status_code}. Continuing anyway.")
    if session is None and not token and username and password:
        print("Logging in via cookie-based auth (user/login)...")
        try:
            session = get_acled_session(username, password)
            print("Session started. Using cookie auth for all requests.")
        except Exception as e:
            print(f"Cookie login failed: {e}")
            raise SystemExit(1) from e
    if session is None and not token:
        print("Set ACLED_ACCESS_TOKEN or ACLED_USERNAME + ACLED_PASSWORD in .env")
        raise SystemExit(1)

    data_dir = _data_dir()
    country_names = list(COUNTRIES.values())

    try:
        for i, country in enumerate(country_names):
            try:
                print(f"  Fetching {country}...", end=" ", flush=True)
                df = fetch_acled(country, token=token, session=session)
                n = len(df)
                out_path = data_dir / f"{country.lower().replace(' ', '_')}.csv"
                if n > 0:
                    df.to_csv(out_path, index=False)
                print(f"{n} events -> {out_path.name}")
            except requests.RequestException as e:
                print(f"  {country}: failed - {e}")
            if i < len(country_names) - 1:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)

    # Sample features for Ukraine
    ukraine_csv = data_dir / "ukraine.csv"
    if ukraine_csv.exists():
        df_ua = pd.read_csv(ukraine_csv)
        features = compute_acled_features(df_ua)
        print("Sample compute_acled_features(Ukraine):", features)
    else:
        print("Ukraine CSV not found; skipping feature sample.")
