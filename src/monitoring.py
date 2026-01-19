# src/monitoring.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from src.feature_engineering import (
    num_features,
    cat_features,
    TARGET_DISR,
    TARGET_CONG,
    PROCESSED_PATH,
)
from src.train_model import MODEL_PATH

# Dataset de référence (train)
BASE_TRAIN_PATH = PROCESSED_PATH

# Log de prod (à alimenter en continu si possible)
PROD_LOG_PATH = Path("data/prod_predictions.parquet")


# -----------------------------
# Chargement des données
# -----------------------------

def load_reference_data() -> pd.DataFrame:
    """
    Charge le dataset d'entraînement comme référence.
    """
    return pd.read_parquet(BASE_TRAIN_PATH)


def load_current_data(sample_path: Path | None = None) -> pd.DataFrame:
    """
    Charge un dataset "courant" à comparer (par ex. dernier batch).
    Si path non fourni, réutilise BASE_TRAIN_PATH comme mock.
    """
    path = sample_path or BASE_TRAIN_PATH
    return pd.read_parquet(path)


# -----------------------------
# Data drift (numérique / catégoriel)
# -----------------------------

def _numeric_drift_summary(ref: pd.Series, cur: pd.Series) -> dict:
    """
    Petit résumé de drift pour une feature numérique :
    diff de moyenne, diff d'écart-type, KS-like sur quantiles.
    """
    ref = ref.dropna()
    cur = cur.dropna()
    if ref.empty or cur.empty:
        return {"mean_ref": np.nan, "mean_cur": np.nan, "delta_mean": np.nan}

    mean_ref = ref.mean()
    mean_cur = cur.mean()
    delta_mean = mean_cur - mean_ref

    std_ref = ref.std()
    std_cur = cur.std()

    # Différence moyenne sur quantiles comme proxy de drift
    qs = np.linspace(0.1, 0.9, 9)
    ref_q = ref.quantile(qs).values
    cur_q = cur.quantile(qs).values
    quant_diff = float(np.mean(np.abs(ref_q - cur_q)))

    return {
        "mean_ref": float(mean_ref),
        "mean_cur": float(mean_cur),
        "delta_mean": float(delta_mean),
        "std_ref": float(std_ref),
        "std_cur": float(std_cur),
        "quant_diff": quant_diff,
    }


def _categorical_drift_summary(ref: pd.Series, cur: pd.Series) -> dict:
    """
    Drift simple pour une variable catégorielle :
    compare les distributions de fréquence.
    """
    ref_counts = (ref.value_counts(normalize=True)).to_dict()
    cur_counts = (cur.value_counts(normalize=True)).to_dict()

    # union des catégories
    cats = set(ref_counts.keys()) | set(cur_counts.keys())
    l1_dist = 0.0
    for c in cats:
        l1_dist += abs(ref_counts.get(c, 0.0) - cur_counts.get(c, 0.0))

    return {
        "l1_dist": float(l1_dist),
        "top_ref": ref_counts,
        "top_cur": cur_counts,
    }


def compute_data_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame) -> dict:
    """
    Retourne un dict avec un résumé de drift pour quelques features clés.
    """
    drift = {"numeric": {}, "categorical": {}}

    for col in num_features:
        if col in ref_df.columns and col in cur_df.columns:
            drift["numeric"][col] = _numeric_drift_summary(ref_df[col], cur_df[col])

    for col in cat_features:
        if col in ref_df.columns and col in cur_df.columns:
            drift["categorical"][col] = _categorical_drift_summary(ref_df[col], cur_df[col])

    return drift


# -----------------------------
# Perf récente (prod)
# -----------------------------

def compute_recent_performance() -> dict:
    """
    Perf récente si on a logué les prédictions et les vrais labels
    dans PROD_LOG_PATH (à toi de remplir ce fichier plus tard).
    """
    if not PROD_LOG_PATH.exists():
        return {"available": False, "message": "No production log file found."}

    df = pd.read_parquet(PROD_LOG_PATH)

    if "y_true" not in df.columns or "y_pred" not in df.columns:
        return {"available": False, "message": "Production log missing y_true / y_pred."}

    from sklearn.metrics import f1_score, accuracy_score

    y_true = df["y_true"]
    y_pred = df["y_pred"]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    return {
        "available": True,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "n_samples": int(len(df)),
    }


# -----------------------------
# Insights opérationnels STL
# -----------------------------

def compute_operational_insights(df: pd.DataFrame) -> dict:
    """
    Quelques insights prêts à afficher :
    - compagnies les plus en retard
    - routes les plus à risque
    - créneaux horaires les plus critiques
    - profil météo
    """
    insights: dict = {}

    # 1) Compagnies en retard (ArrDelay > 15)
    if "ArrDelay" in df.columns and "Airline" in df.columns:
        df_tmp = df.copy()
        df_tmp["is_delayed_15"] = (df_tmp["ArrDelay"] > 15).astype(int)
        comp = (
            df_tmp.groupby("Airline")
            .agg(
                avg_arr_delay=("ArrDelay", "mean"),
                pct_delayed_15=("is_delayed_15", "mean"),
                n_flights=("ArrDelay", "count"),
            )
            .sort_values("pct_delayed_15", ascending=False)
        )
        comp["pct_delayed_15"] = (comp["pct_delayed_15"] * 100).round(1)
        comp["avg_arr_delay"] = comp["avg_arr_delay"].round(1)
        insights["airline_delays"] = comp.reset_index()

    # 2) Routes à risque (Origin–Dest) en termes de disruption_risk
    if TARGET_DISR in df.columns and "Dest" in df.columns and "Origin" in df.columns:
        route_risk = (
            df.groupby(["Origin", "Dest", TARGET_DISR])
            .size()
            .reset_index(name="n")
        )
        # Part de HIGH_DISRUPTION par route
        total_route = route_risk.groupby(["Origin", "Dest"])["n"].transform("sum")
        route_risk["share"] = route_risk["n"] / total_route
        high = route_risk[route_risk[TARGET_DISR] == "HIGH_DISRUPTION"].copy()
        high = high.sort_values("share", ascending=False)
        insights["route_high_risk"] = high.reset_index(drop=True)

    # 3) Créneaux horaires critiques (DepHour) sur le risque de disruption
    if "DepHour" in df.columns and TARGET_DISR in df.columns:
        hour_risk = (
            df.groupby(["DepHour", TARGET_DISR])
            .size()
            .reset_index(name="n")
        )
        total_hour = hour_risk.groupby("DepHour")["n"].transform("sum")
        hour_risk["share"] = hour_risk["n"] / total_hour
        high_h = hour_risk[hour_risk[TARGET_DISR] == "HIGH_DISRUPTION"].copy()
        high_h = high_h.sort_values("share", ascending=False)
        insights["hour_high_risk"] = high_h.reset_index(drop=True)

    # 4) Météo : fréquence des wx_risk_score élevés
    if "wx_risk_score" in df.columns:
        wx = df.copy()
        wx["wx_high"] = (wx["wx_risk_score"] > 0.6).astype(int)
        wx_stats = wx["wx_high"].mean() * 100
        insights["weather_high_share"] = float(round(wx_stats, 1))

    return insights


# -----------------------------
# Script de test / debug
# -----------------------------

def main():
    # Dataset de référence (train)
    ref_df = load_reference_data()
    # Pour l'instant, comme "courant", on reprend le même jeu (mock)
    cur_df = load_current_data()

    drift = compute_data_drift(ref_df, cur_df)
    perf = compute_recent_performance()
    insights = compute_operational_insights(ref_df)

    print("Data drift summary (numeric features):")
    for col, stats in drift["numeric"].items():
        print(f"- {col}: Δmean={stats['delta_mean']:.3f}, quant_diff={stats['quant_diff']:.3f}")

    print("\nData drift summary (categorical features):")
    for col, stats in drift["categorical"].items():
        print(f"- {col}: L1 distance={stats['l1_dist']:.3f}")

    print("\nRecent performance:")
    print(perf)

    print("\nOperational insights (top compagnies en retard):")
    if "airline_delays" in insights:
        print(insights["airline_delays"].head(5))

    print("\nRoutes avec forte part de HIGH_DISRUPTION:")
    if "route_high_risk" in insights:
        print(insights["route_high_risk"].head(5))

    print("\nCréneaux horaires avec forte part de HIGH_DISRUPTION:")
    if "hour_high_risk" in insights:
        print(insights["hour_high_risk"].head(5))

    if "weather_high_share" in insights:
        print(f"\nPart de vols avec wx_risk_score>0.6 : {insights['weather_high_share']:.1f}%")

if __name__ == "__main__":
    main()
