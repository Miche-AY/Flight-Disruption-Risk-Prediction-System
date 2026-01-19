# src/feature_engineering.py
import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/Flight_delay.csv")
PROCESSED_PATH = Path("data/processed_flights.parquet")

# listes exposées pour le modèle / app
num_features = [
    "DepHour",
    "ArrHour",
    "Month",
    "DayOfWeek",
    "IsWeekend",
    "IsPeakHour",
    "Distance",
    "flights_origin_hour",
    "airport_congestion_score",
    "route_mean_delay",
    "route_delay_std",
    "additional_taxiout",
    "turnaround_prev",
    "late_inbound_flag",
    "short_turnaround_flag",
    "carrier_delay_rate",
    "route_cancel_rate",
    "DelayRiskIndex",
    # météo dérivée
    "wx_delay_share",
    "wx_has_history",
    "wx_route_delay_rate",
    "wx_seasonality",
    "wx_risk_score",
    # nouvelles features congestion
    "taxiout_excess",
    "taxiin_excess",
    "origin_hourly_density",
    "dest_hourly_density",
    "nas_delay_flag",
    "system_congestion_score",
    "congestion_score",
]

cat_features = [
    "Origin",
    "Dest",
    "Org_Airport",      # nom complet origine
    "Dest_Airport",     # nom complet destination
    "UniqueCarrier",
    "Airline",
    "DepSlot",
]

TARGET_CONG = "congestion_risk"
TARGET_DISR = "disruption_risk"


def parse_hhmm_to_hour(x):
    """Convertit un champ hhmm (int/str) en heure (0-23)."""
    if pd.isna(x):
        return np.nan
    s = str(int(x)).zfill(4)
    try:
        return int(s[:2])
    except Exception:
        return np.nan


def normalize_series(x: pd.Series) -> pd.Series:
    """Normalise une série entre 0 et 1."""
    if x.max() == x.min():
        return pd.Series(0.5, index=x.index)
    return (x - x.min()) / (x.max() - x.min())


def build_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit les features de congestion basées sur des indicateurs structurels.
    """
    df = df.copy()
    
    # 1. Congestion au départ (STL) via TaxiOut
    if "TaxiOut" in df.columns:
        taxiout_baseline = df["TaxiOut"].quantile(0.25)
        df["taxiout_excess"] = (df["TaxiOut"] - taxiout_baseline).clip(lower=0)
    else:
        df["taxiout_excess"] = 0.0
    
    # 2. Densité horaire à l'origine (STL)
    df["origin_hourly_density"] = (
        df.groupby(["Date", "DepHour"])["FlightNum"]
        .transform("count")
        .astype(float)
    )
    
    # 3. Congestion à l'arrivée via TaxiIn
    if "TaxiIn" in df.columns:
        taxiin_baseline_by_dest = (
            df.groupby("Dest")["TaxiIn"]
            .transform(lambda x: x.quantile(0.25))
        )
        df["taxiin_excess"] = (df["TaxiIn"] - taxiin_baseline_by_dest).clip(lower=0)
    else:
        df["taxiin_excess"] = 0.0
    
    # 4. Densité horaire à destination
    if "ArrHour" in df.columns:
        df["dest_hourly_density"] = (
            df.groupby(["Dest", "Date", "ArrHour"])["FlightNum"]
            .transform("count")
            .astype(float)
        )
    else:
        df["dest_hourly_density"] = 0.0
    
    # 5. Congestion système (NAS delays)
    if "NASDelay" in df.columns:
        df["nas_delay_flag"] = (df["NASDelay"].fillna(0) > 0).astype(int)
        df["system_congestion_score"] = (
            df.groupby(["Date", "DepHour"])["nas_delay_flag"]
            .transform("mean")
            .fillna(0)
        )
    else:
        df["nas_delay_flag"] = 0
        df["system_congestion_score"] = 0.0
    
    # 6. Score composite de congestion
    df["congestion_score"] = (
        0.25 * normalize_series(df["taxiout_excess"]) +
        0.20 * normalize_series(df["taxiin_excess"]) +
        0.20 * normalize_series(df["origin_hourly_density"]) +
        0.15 * normalize_series(df["dest_hourly_density"]) +
        0.20 * df["system_congestion_score"]
    ).clip(0, 1)
    
    return df


def build_congestion_target(df: pd.DataFrame) -> pd.Series:
    """
    Cible congestion AMÉLIORÉE basée sur des indicateurs structurels.
    
    Utilise :
    - Score composite de congestion (TaxiOut/In, densité, NAS)
    - Annulations et détournements
    - Retards NAS significatifs
    """
    # Récupérer les variables nécessaires
    cancelled = df.get("Cancelled", 0).fillna(0)
    diverted = df.get("Diverted", 0).fillna(0)
    nas_delay = df.get("NASDelay", 0).fillna(0)
    
    # S'assurer que congestion_score existe
    if "congestion_score" not in df.columns:
        raise ValueError("congestion_score doit être calculé avant build_congestion_target")
    
    # Définir les seuils dynamiques basés sur la distribution
    low_threshold = df["congestion_score"].quantile(0.60)
    high_threshold = df["congestion_score"].quantile(0.85)
    
    # Conditions de congestion élevée
    is_high = (
        (df["congestion_score"] > high_threshold) |
        (cancelled == 1) |
        (diverted == 1) |
        (nas_delay > 30)  # NAS delay > 30 min = congestion système
    )
    
    # Conditions de congestion modérée
    is_mid = (
        (df["congestion_score"] > low_threshold) &
        (~is_high)
    )
    
    return pd.Series(
        np.where(
            is_high,
            "CONGESTION_PROBABLE",
            np.where(is_mid, "RISK_CONGESTION", "NO_CONGESTION"),
        ),
        index=df.index,
    )


def build_disruption_target(df: pd.DataFrame) -> pd.Series:
    """
    Cible 'risque de disruption opérationnelle':
    - NO_DISRUPTION   : ArrDelay ≤ 15 et pas annulé/détourné
    - RISK_DISRUPTION : 15 < ArrDelay ≤ 45
    - HIGH_DISRUPTION : ArrDelay > 45 ou Cancelled=1 ou Diverted=1
    """
    cancelled = df.get("Cancelled", 0).fillna(0)
    diverted = df.get("Diverted", 0).fillna(0)
    arr_delay = df["ArrDelay"].fillna(0)

    is_high = (arr_delay > 45) | (cancelled == 1) | (diverted == 1)
    is_mid = (arr_delay > 15) & (~is_high)

    return pd.Series(
        np.where(
            is_high,
            "HIGH_DISRUPTION",
            np.where(is_mid, "RISK_DISRUPTION", "NO_DISRUPTION"),
        ),
        index=df.index,
    )


def engineer_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Utilise WeatherDelay et l'historique pour construire des indicateurs météo.
    Hypothèse: WeatherDelay>0 signifie que la météo a contribué au retard.
    """
    df = df.copy()

    if "WeatherDelay" not in df.columns:
        df["wx_delay_share"] = 0.0
        df["wx_has_history"] = 0.0
        df["wx_route_delay_rate"] = 0.0
        df["wx_seasonality"] = 0.5
        df["wx_risk_score"] = 0.3
        return df

    total_delay = df["ArrDelay"].clip(lower=0)
    wx_delay = df["WeatherDelay"].fillna(0).clip(lower=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(total_delay > 0, wx_delay / (total_delay + 1e-6), 0.0)
    df["wx_delay_share"] = np.clip(share, 0, 1)

    df["wx_event"] = (df["WeatherDelay"] > 0).astype(int)
    route_wx_rate = (
        df.groupby(["Origin", "Dest"])["wx_event"]
        .transform("mean")
        .fillna(0)
    )
    df["wx_route_delay_rate"] = route_wx_rate

    if "Month" in df.columns:
        month_wx_rate = (
            df.groupby("Month")["wx_event"]
            .transform("mean")
            .fillna(0)
        )
        df["wx_seasonality"] = month_wx_rate
    else:
        df["wx_seasonality"] = 0.5

    df["wx_has_history"] = (df["wx_route_delay_rate"] > 0.05).astype(int)

    df["wx_risk_score"] = (
        0.4 * df["wx_delay_share"]
        + 0.3 * df["wx_route_delay_rate"]
        + 0.3 * df["wx_seasonality"]
    ).clip(0, 1)

    return df


def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # --- Filtrer STL uniquement ---
    df = df[df["Origin"] == "STL"].copy()

    # --- Temps / saison ---
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["DayOfWeek"].astype(int)

    df["DepHour"] = df["DepTime"].apply(parse_hhmm_to_hour)
    df["ArrHour"] = df["ArrTime"].apply(parse_hhmm_to_hour)

    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)
    df["IsPeakHour"] = df["DepHour"].isin([7, 8, 9, 10, 17, 18, 19, 20]).astype(int)

    df["DepSlot"] = np.select(
        [
            (df["DepHour"] >= 6) & (df["DepHour"] < 12),
            (df["DepHour"] >= 12) & (df["DepHour"] < 18),
            (df["DepHour"] >= 18) & (df["DepHour"] < 24),
        ],
        ["MORNING", "AFTERNOON", "EVENING"],
        default="NIGHT",
    )

    # --- Trafic / congestion STL ---
    flights_origin_hour = (
        df.groupby(["Origin", "Date", "DepHour"])["FlightNum"]
        .transform("count")
        .astype(float)
    )
    df["flights_origin_hour"] = flights_origin_hour

    df["airport_congestion_score"] = normalize_series(df["flights_origin_hour"])

    route_stats = (
        df.groupby(["Origin", "Dest"])["ArrDelay"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "route_mean_delay", "std": "route_delay_std"})
    )
    df = df.merge(route_stats, on=["Origin", "Dest"], how="left")

    # --- Taxi-out supplémentaire ---
    if "TaxiOut" in df.columns:
        base_taxi = (
            df[["Origin", "TaxiOut"]]
            .dropna()
            .groupby("Origin")["TaxiOut"]
            .quantile(0.1)
            .rename("taxiout_baseline")
        )
        df = df.merge(base_taxi, on="Origin", how="left")
        df["additional_taxiout"] = (df["TaxiOut"] - df["taxiout_baseline"]).fillna(0)
    else:
        df["additional_taxiout"] = 0.0

    # --- Rotation appareil / chaîne vols ---
    df = df.sort_values(["TailNum", "Date", "DepTime"])

    df["PrevArrTime"] = df.groupby("TailNum")["ArrTime"].shift(1)
    df["PrevDate"] = df.groupby("TailNum")["Date"].shift(1)

    def to_minutes(date_col, time_col):
        hour = time_col.apply(parse_hhmm_to_hour)
        minute = (time_col.fillna(0).astype(int) % 100).astype(int)
        return (date_col.view("int64") // 10**9 // 60) + hour * 60 + minute

    valid_prev = df["PrevDate"].notna()
    turnaround_prev = pd.Series(np.nan, index=df.index)
    if valid_prev.any():
        arr_prev_min = to_minutes(df.loc[valid_prev, "PrevDate"], df.loc[valid_prev, "PrevArrTime"])
        dep_min = to_minutes(df.loc[valid_prev, "Date"], df.loc[valid_prev, "DepTime"])
        turnaround_prev.loc[valid_prev] = dep_min.values - arr_prev_min.values

    df["turnaround_prev"] = turnaround_prev
    df["late_inbound_flag"] = (
        df.groupby("TailNum")["ArrDelay"].shift(1).fillna(0) > 30
    ).astype(int)
    df["short_turnaround_flag"] = (df["turnaround_prev"] < 45).fillna(0).astype(int)

    # --- Fiabilité compagnie / route ---
    df["is_delayed_15"] = (df["ArrDelay"] > 15).astype(int)
    carrier_delay_rate = (
        df.groupby("UniqueCarrier")["is_delayed_15"].transform("mean")
    )
    df["carrier_delay_rate"] = carrier_delay_rate

    if "Cancelled" in df.columns:
        route_cancel_rate = (
            df.groupby(["Origin", "Dest"])["Cancelled"].transform("mean")
        )
        df["route_cancel_rate"] = route_cancel_rate
    else:
        df["route_cancel_rate"] = 0.0

    for col in ["route_mean_delay", "carrier_delay_rate", "route_cancel_rate"]:
        df[col] = df[col].fillna(df[col].median())

    df["DelayRiskIndex"] = (
        normalize_series(df["route_mean_delay"]) * 0.5
        + normalize_series(df["carrier_delay_rate"]) * 0.3
        + normalize_series(df["route_cancel_rate"]) * 0.2
    )

    # --- Météo dérivée à partir de WeatherDelay ---
    df = engineer_weather_features(df)

    # --- NOUVELLES FEATURES DE CONGESTION ---
    df = build_congestion_features(df)

    # --- Cibles multi-classes ---
    df[TARGET_CONG] = build_congestion_target(df)
    df[TARGET_DISR] = build_disruption_target(df)

    # Colonnes finales (en gardant ArrDelay pour le dashboard)
    base_cols = [TARGET_CONG, TARGET_DISR, "ArrDelay"] + num_features + cat_features
    base_cols = [c for c in base_cols if c in df.columns]

    df_final = df[base_cols].dropna(subset=[TARGET_DISR]).reset_index(drop=True)
    return df_final


def main():
    print("Loading raw data from", RAW_PATH)
    df_raw = pd.read_csv(RAW_PATH)

    print("Engineering features for STL departures...")
    df_proc = engineer_features(df_raw)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_proc.to_parquet(PROCESSED_PATH, index=False)
    print("Saved processed data to", PROCESSED_PATH)
    print("Shape:", df_proc.shape)
    print("\n" + "="*60)
    print("CONGESTION TARGET DISTRIBUTION (nouvelle version)")
    print("="*60)
    print(df_proc[TARGET_CONG].value_counts(normalize=True).sort_index())
    print("\n" + "="*60)
    print("DISRUPTION TARGET DISTRIBUTION")
    print("="*60)
    print(df_proc[TARGET_DISR].value_counts(normalize=True).sort_index())
    
    # Statistiques sur les features de congestion
    print("\n" + "="*60)
    print("CONGESTION FEATURES STATISTICS")
    print("="*60)
    congestion_cols = [
        "taxiout_excess", "taxiin_excess", "origin_hourly_density",
        "dest_hourly_density", "system_congestion_score", "congestion_score"
    ]
    print(df_proc[congestion_cols].describe())


if __name__ == "__main__":
    main()