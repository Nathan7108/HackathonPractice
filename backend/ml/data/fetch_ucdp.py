# Sentinel AI — load UCDP conflict data (S1-04)
# Data from local CSV files in data/ucdp/ (manually downloaded; API not used).

import sys
from pathlib import Path

import pandas as pd

# ISO2 -> display name for monitored countries
COUNTRIES = {
    "UA": "Ukraine",
    "TW": "Taiwan",
    "IR": "Iran (Government of)",
    "VE": "Venezuela",
    "PK": "Pakistan",
    "ET": "Ethiopia",
    "RS": "Serbia",
    "BR": "Brazil",
}

# GED CSV uses "country" column; some names differ from COUNTRIES
COUNTRY_NAME_IN_GED = {
    "Iran (Government of)": "Iran",
}


def _data_dir() -> Path:
    """Project data dir: data/ucdp/ (from repo root)."""
    root = Path(__file__).resolve().parents[3]
    d = root / "data" / "ucdp"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ged_csv_path() -> Path | None:
    """Path to the GED CSV in data/ucdp/ (e.g. GEDEvent_v25_1.csv)."""
    data_dir = _data_dir()
    for name in ["GEDEvent_v25_1.csv", "ged251.csv"]:
        p = data_dir / name
        if p.exists():
            return p
    for p in data_dir.glob("GED*.csv"):
        return p
    return None


def _armed_conflict_csv_path() -> Path | None:
    """Path to the UCDP/PRIO Armed Conflict CSV in data/ucdp/."""
    data_dir = _data_dir()
    for name in ["UcdpPrioConflict_v25_1.csv", "ucdp-prio-acd-251.csv", "armed_conflict_1946_2023.csv"]:
        p = data_dir / name
        if p.exists():
            return p
    for p in data_dir.glob("*[Pp]rio**.csv"):
        return p
    return None


def load_ucdp_ged(country_name: str) -> pd.DataFrame:
    """
    Load UCDP GED for one country from the local GED CSV in data/ucdp/.
    Filters by country name (column 'country'). Returns DataFrame with columns
    including year, country, deaths_a, deaths_b, deaths_civilians, type_of_violence,
    latitude, longitude.
    """
    path = _ged_csv_path()
    if path is None:
        return pd.DataFrame()

    name_in_csv = COUNTRY_NAME_IN_GED.get(country_name, country_name)
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    if "country" not in df.columns:
        return pd.DataFrame()
    return df.loc[df["country"] == name_in_csv].copy()


def load_ucdp_armed_conflict() -> pd.DataFrame:
    """
    Load UCDP/PRIO Armed Conflict Dataset from the local CSV in data/ucdp/.
    Returns full DataFrame (1946–2024).
    """
    path = _armed_conflict_csv_path()
    if path is None:
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")


def compute_ucdp_features(ged_df: pd.DataFrame, window_years: int = 5) -> dict:
    """
    Compute 5 UCDP features per ML Guide Section 2.3.
    Returns dict with exactly: ucdp_total_deaths, ucdp_state_conflict_years,
    ucdp_civilian_deaths, ucdp_conflict_intensity, ucdp_recurrence_rate.
    """
    keys = [
        "ucdp_total_deaths",
        "ucdp_state_conflict_years",
        "ucdp_civilian_deaths",
        "ucdp_conflict_intensity",
        "ucdp_recurrence_rate",
    ]
    if ged_df is None or len(ged_df) == 0:
        return {k: 0 for k in keys}

    ged_df = ged_df.copy()
    ged_df["year"] = pd.to_numeric(ged_df.get("year", 0), errors="coerce")
    ged_df = ged_df.dropna(subset=["year"])
    if len(ged_df) == 0:
        return {k: 0 for k in keys}

    max_year = ged_df["year"].max()
    recent = ged_df[ged_df["year"] >= max_year - window_years]
    if len(recent) == 0:
        return {k: 0 for k in keys}

    type_col = recent.get("type_of_violence", pd.Series(dtype=object))
    type_str = type_col.astype(str)
    state_based = recent[type_str == "1"]
    civilian = recent[type_str == "3"]

    deaths_a = recent.get("deaths_a", pd.Series(0)).fillna(0)
    deaths_b = recent.get("deaths_b", pd.Series(0)).fillna(0)
    deaths_civ = civilian.get("deaths_civilians", pd.Series(0)).fillna(0)

    return {
        "ucdp_total_deaths": float(deaths_a.sum() + deaths_b.sum()),
        "ucdp_state_conflict_years": (
            len(state_based["year"].unique()) if len(state_based) > 0 else 0
        ),
        "ucdp_civilian_deaths": float(deaths_civ.sum()),
        "ucdp_conflict_intensity": float(deaths_a.mean()),
        "ucdp_recurrence_rate": float(len(recent["year"].unique()) / max(window_years, 1)),
    }


def build_ucdp_training_labels(armed_conflict_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build labeled country-years for XGBoost training from armed conflict dataset.
    Returns DataFrame with columns: location, year, conflict_id, intensity_level, risk_label.
    risk_label is LOW / ELEVATED / HIGH. Prints value counts.
    """
    df = armed_conflict_df.copy()

    # UCDP/PRIO CSV often uses: loc (or gwnoloc), year, conflictid, intensity
    rename = {}
    if "loc" in df.columns:
        rename["loc"] = "location"
    if "gwno_loc" in df.columns and "location" not in df.columns:
        rename["gwno_loc"] = "location"
    if "gwnoloc" in df.columns and "location" not in df.columns:
        rename["gwnoloc"] = "location"
    if "conflictid" in df.columns:
        rename["conflictid"] = "conflict_id"
    if "intensity" in df.columns:
        rename["intensity"] = "intensity_level"
    df = df.rename(columns=rename)

    def assign_label(row):
        cid = row.get("conflict_id", row.get("conflictid", None))
        if pd.isna(cid) or cid == "" or (isinstance(cid, (int, float)) and cid == 0):
            return "LOW"
        intensity = row.get("intensity_level", row.get("intensity", 0))
        try:
            intensity = int(float(intensity))
        except (ValueError, TypeError):
            intensity = 0
        if intensity >= 2:
            return "HIGH"
        if intensity == 1:
            return "ELEVATED"
        return "LOW"

    df["risk_label"] = df.apply(assign_label, axis=1)

    out_cols = ["location", "year", "conflict_id", "intensity_level", "risk_label"]
    available = [c for c in out_cols if c in df.columns]
    result = df[available].copy()

    print(f"Labeled {len(result)} country-years")
    print(result["risk_label"].value_counts())
    return result


if __name__ == "__main__":
    data_dir = _data_dir()

    # List extracted CSVs and their shapes
    print("UCDP local CSVs in data/ucdp/:")
    for p in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(p, nrows=0)
        n = sum(1 for _ in open(p, encoding="utf-8", errors="ignore")) - 1
        print(f"  {p.name}: shape=({n}, {len(df.columns)})")

    country_names = list(COUNTRIES.values())

    print("\nLoading GED by country:")
    for country in country_names:
        df = load_ucdp_ged(country)
        print(f"  {country}: {len(df)} events")

    print("\nLoading UCDP/PRIO Armed Conflict Dataset...")
    ac_df = load_ucdp_armed_conflict()
    print(f"  Armed conflict rows: {len(ac_df)}")
    if len(ac_df) > 0 and "year" in ac_df.columns:
        print(f"  Year range: {ac_df['year'].min()} - {ac_df['year'].max()}")

    # Sample compute_ucdp_features for Ukraine
    df_ua = load_ucdp_ged("Ukraine")
    if len(df_ua) > 0:
        features = compute_ucdp_features(df_ua)
        print("\nSample compute_ucdp_features(Ukraine):", features)
    else:
        print("\nNo Ukraine GED data; skipping feature sample.")

    # Training label distribution
    if len(ac_df) > 0:
        print("\nTraining labels from build_ucdp_training_labels():")
        labels_df = build_ucdp_training_labels(ac_df)
        print("Done.")
