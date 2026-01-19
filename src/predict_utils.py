# src/predict_utils.py
import os
import json
import numpy as np
import pandas as pd
import requests

# -----------------------------
# Données historiques pour le chat
# -----------------------------

DATA_PATH = "data/processed_flights.parquet"
_df_stats = None


def _get_df_stats() -> pd.DataFrame:
    """
    Charge une seule fois les données historiques pour le chat / stats.
    """
    global _df_stats
    if _df_stats is None:
        _df_stats = pd.read_parquet(DATA_PATH)
    return _df_stats


def _stats_delays_by_company(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stats retards par compagnie (Airline):
    - retard moyen d'arrivée
    - % de vols >15 min
    - nb de vols
    """
    df_tmp = df.copy()
    if "ArrDelay" not in df_tmp.columns:
        return pd.DataFrame()

    df_tmp["is_delayed_15"] = (df_tmp["ArrDelay"] > 15).astype(int)
    agg = (
        df_tmp.groupby("Airline")
        .agg(
            avg_arr_delay=("ArrDelay", "mean"),
            pct_delayed_15=("is_delayed_15", "mean"),
            n_flights=("ArrDelay", "count"),
        )
        .sort_values("pct_delayed_15", ascending=False)
    )
    agg["pct_delayed_15"] = (agg["pct_delayed_15"] * 100).round(1)
    agg["avg_arr_delay"] = agg["avg_arr_delay"].round(1)
    return agg


def _stats_delays_by_carrier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stats retards par code compagnie (UniqueCarrier).
    """
    df_tmp = df.copy()
    if "ArrDelay" not in df_tmp.columns:
        return pd.DataFrame()

    df_tmp["is_delayed_15"] = (df_tmp["ArrDelay"] > 15).astype(int)
    agg = (
        df_tmp.groupby("UniqueCarrier")
        .agg(
            avg_arr_delay=("ArrDelay", "mean"),
            pct_delayed_15=("is_delayed_15", "mean"),
            n_flights=("ArrDelay", "count"),
        )
        .sort_values("pct_delayed_15", ascending=False)
    )
    agg["pct_delayed_15"] = (agg["pct_delayed_15"] * 100).round(1)
    agg["avg_arr_delay"] = agg["avg_arr_delay"].round(1)
    return agg


# -----------------------------
# Mapping des classes de risque
# -----------------------------

DISPLAY_RISK_MAP = {
    "NO_DISRUPTION": "Pas de risque de disruption",
    "RISK_DISRUPTION": "Risque de disruption",
    "HIGH_DISRUPTION": "Disruption probable",
}

DISPLAY_CONG_MAP = {
    "NO_CONGESTION": "Pas de congestion",
    "RISK_CONGESTION": "Risque de congestion",
    "CONGESTION_PROBABLE": "Congestion probable",
}


# -----------------------------
# Approximation du risque de congestion
# -----------------------------

def derive_congestion_label(sample_row: pd.Series) -> str:
    """
    Approximation du niveau de congestion à partir des features
    (utile si on ne passe pas la vraie 'congestion_risk' depuis le dataset).
    """
    cong_score = sample_row.get("airport_congestion_score", 0)
    taxi_add = sample_row.get("additional_taxiout", 0)

    if cong_score > 0.7 or taxi_add > 10:
        return "CONGESTION_PROBABLE"
    if cong_score > 0.4 or taxi_add > 5:
        return "RISK_CONGESTION"
    return "NO_CONGESTION"


# -----------------------------
# Règles métier / Actions OCC
# -----------------------------

def recommend_actions(
    sample_row: pd.Series,
    disruption_label: str,
    proba_dict: dict,
    congestion_label: str | None = None,
) -> list[str]:
    """
    Règles métier pour STL, combinant risque de DISRUPTION et niveau de CONGESTION.
    """
    actions: list[str] = []

    # Congestion : soit label explicite, soit dérivé
    if congestion_label is None:
        congestion_label = derive_congestion_label(sample_row)

    # 0) Synthèse risques
    synthese = (
        f"Risque de disruption : {DISPLAY_RISK_MAP.get(disruption_label, disruption_label)} "
        f"(NO={proba_dict.get('NO_DISRUPTION', 0):.2f}, "
        f"RISK={proba_dict.get('RISK_DISRUPTION', 0):.2f}, "
        f"HIGH={proba_dict.get('HIGH_DISRUPTION', 0):.2f}). "
        f"Niveau de congestion estimé : {DISPLAY_CONG_MAP.get(congestion_label, congestion_label)}."
    )
    actions.append(synthese)

    # 1) Flux / capacité aéroport
    if congestion_label == "CONGESTION_PROBABLE":
        actions.append(
            "Flux – Congestion probable à STL : décaler ou séquencer certains départs/arrivées autour de ce vol "
            "pour lisser la charge piste et gates, et coordonner les slots avec l'ATC."
        )
    elif congestion_label == "RISK_CONGESTION":
        actions.append(
            "Flux – Risque de congestion : vérifier les plans de gate, éviter les regroupements de départs sur le même créneau "
            "et préparer des gates alternatifs pour absorber les aléas."
        )

    # 2) Appareil / rotation / crew
    if (sample_row.get("late_inbound_flag", 0) == 1) or (
        sample_row.get("short_turnaround_flag", 0) == 1
    ):
        actions.append(
            "Appareil/Crew – Rotation fragile : envisager la réaffectation d'appareil ou de crew, "
            "ou allonger le temps de rotation planifié pour limiter le retard propagé."
        )

    if sample_row.get("additional_taxiout", 0) > 5:
        actions.append(
            "Appareil/Crew – Taxi-out supérieur à la normale sur STL : anticiper un temps de roulage long et "
            "coordonner avec l'ATC pour réduire l'attente en file."
        )

    # 3) Passagers / coûts
    if sample_row.get("Distance", 0) > 1500 and disruption_label in ["RISK_DISRUPTION", "HIGH_DISRUPTION"]:
        actions.append(
            "Passagers – Vol long-courrier : anticiper l'impact sur les correspondances internationales et "
            "les obligations de compensation potentielles."
        )

    if sample_row.get("DelayRiskIndex", 0) > 0.7 and disruption_label == "HIGH_DISRUPTION":
        actions.append(
            "Coûts – Vol à fort risque global (DelayRiskIndex élevé) : préparer des options de reroutage à moindre coût "
            "et un budget de mitigation (rebooking, handling, indemnisation)."
        )

    # 4) Météo
    if sample_row.get("wx_risk_score", 0) > 0.6 or sample_row.get("wx_route_delay_rate", 0) > 0.5:
        actions.append(
            "Météo – Risque météo significatif (historique de retards météo sur cet axe) : coordonner avec MET/ATC, "
            "prévoir des marges supplémentaires et des slots alternatifs si la capacité se réduit."
        )

    # 5) Monitoring / niveau d’alerte OCC
    if disruption_label == "HIGH_DISRUPTION":
        actions.append(
            "Monitoring – Niveau HIGH : escalader vers le duty manager OCC, suivre ce vol en temps réel "
            "et valider un plan d'action dans les 30 prochaines minutes."
        )
    elif disruption_label == "RISK_DISRUPTION":
        if congestion_label in ["RISK_CONGESTION", "CONGESTION_PROBABLE"]:
            actions.append(
                "Monitoring – Risque de disruption en contexte congestionné : placer le vol en surveillance renforcée "
                "avec revue à T-60 et T-30 minutes avant départ."
            )

    if len(actions) == 1:
        actions.append(
            "Monitoring – Niveau de risque faible et faible congestion : surveiller le vol via les alertes automatiques, "
            "sans action corrective immédiate."
        )

    return actions


# -----------------------------
# Prédiction pour un vol unique
# -----------------------------

def predict_single_flight(bundle: dict, sample_row: pd.Series, congestion_label: str | None = None):
    """
    bundle : modèle sauvegardé (pipeline + label_encoder + X_train/X_test)
    sample_row : Series avec toutes les features (num + cat) pour un vol STL.
    congestion_label : optionnel, si tu veux passer le vrai 'congestion_risk' issu du dataset.
    """
    clf = bundle["pipeline"]
    le = bundle["label_encoder"]

    sample_df = sample_row.to_frame().T

    proba = clf.predict_proba(sample_df)[0]  # array (n_classes,)

    risk_idx = int(np.argmax(proba))
    disruption_label = le.inverse_transform([risk_idx])[0]

    proba_dict = {cls: float(proba[i]) for i, cls in enumerate(le.classes_)}

    actions = recommend_actions(sample_row, disruption_label, proba_dict, congestion_label)

    return disruption_label, proba_dict, actions


# -----------------------------
# Chatbot NLQ (LLM hook)
# -----------------------------

def _call_llm_external(prompt: str, context: dict | None = None) -> str:
    """
    Appel optionnel à un LLM externe (OpenRouter / OpenAI).
    Utilise OPENROUTER_API_KEY ou OPENAI_API_KEY si définie dans l'environnement.
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""

    try:
        payload = {
            "model": "openrouter/auto",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Tu es un assistant pour un centre de contrôle des opérations à l'aéroport STL. "
                        "Explique en français simple le risque de disruption et la congestion des vols, "
                        "les causes probables et les actions recommandées, à partir du contexte fourni. "
                        "Utilise un ton clair et opérationnel."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt + "\n\nContexte JSON:\n" + json.dumps(context or {}, ensure_ascii=False),
                },
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


def occ_chat_answer(question: str, last_prediction: dict | None = None) -> str:
    """
    Chat NLQ pour l'OCC :
    - question : texte de l'utilisateur
    - last_prediction : dict avec risk_label, proba_dict, sample, congestion_label, etc.
    """
    text = question.lower()

    # 1) Réponses data-driven (stats locales) si la question parle de retards / compagnies
    if any(w in text for w in ["compagnie", "airline", "compagnies", "carrier"]):
        df = _get_df_stats()
        stats_airline = _stats_delays_by_company(df)
        if not stats_airline.empty:
            top = stats_airline.head(5)
            lines = [
                "Voici un aperçu des retards par compagnie (départs STL uniquement) :",
                "",
            ]
            for name, row in top.iterrows():
                lines.append(
                    f"- {name} : retard moyen {row['avg_arr_delay']} min, "
                    f"{row['pct_delayed_15']}% des vols avec >15 min de retard "
                    f"(n={int(row['n_flights'])})."
                )
            lines.append(
                "Ces chiffres sont calculés sur ton historique local (data/processed_flights.parquet)."
            )
            return "\n".join(lines)

    if any(w in text for w in ["retard", "delay", "retards", "départs en retard"]):
        df = _get_df_stats()
        if "ArrDelay" in df.columns:
            df_tmp = df.copy()
            df_tmp["is_delayed_15"] = (df_tmp["ArrDelay"] > 15).astype(int)
            global_rate = df_tmp["is_delayed_15"].mean() * 100
            avg_delay = df_tmp["ArrDelay"].mean()
            return (
                "Sur l'historique STL disponible :\n"
                f"- Retard moyen à l'arrivée : {avg_delay:.1f} minutes.\n"
                f"- Part des vols avec >15 minutes de retard : {global_rate:.1f}%.\n\n"
                "Tu peux préciser une compagnie ou un axe pour affiner, par exemple : "
                "\"retards pour Delta\" ou \"retards sur STL–ORD\"."
            )

    # 2) Appel LLM externe pour un comportement type mini-ChatGPT
    llm_ctx = last_prediction or {}
    llm_answer = _call_llm_external(question, context=llm_ctx)
    if llm_answer:
        return llm_answer

    # 3) Fallback rules-based si pas de LLM dispo
    if "pourquoi" in text or "why" in text:
        return (
            "Le modèle estime le risque de disruption à partir de signaux comme la congestion aéroportuaire "
            "(flights_origin_hour, airport_congestion_score), la rotation appareil (late_inbound_flag, "
            "short_turnaround_flag, additional_taxiout), l'historique de performance (DelayRiskIndex) et la météo "
            "dérivée de WeatherDelay. La congestion est approximée séparément via des proxies de charge STL, ce qui permet "
            "de distinguer 'risque de disruption' et 'niveau de congestion'."
        )
    if "météo" in text or "weather" in text:
        return (
            "La météo est prise en compte via des indicateurs dérivés de WeatherDelay : part du retard imputable "
            "à la météo (wx_delay_share), fréquence historique des retards météo par route (wx_route_delay_rate) "
            "et par saison (wx_seasonality), combinés dans un wx_risk_score."
        )
    if "congestion" in text:
        return (
            "Le niveau de congestion est estimé via la densité de vols au départ de STL sur chaque créneau horaire "
            "(flights_origin_hour) et un score normalisé airport_congestion_score, complétés par l'additional_taxiout. "
            "Ces signaux alimentent un label de congestion séparé (NO/RISK/PROBABLE) affiché dans l'interface."
        )
    if "actions" in text or "recommand" in text:
        return (
            "Les actions sont structurées par thèmes : Flux (gestion de la congestion), Appareil/Crew (rotation), "
            "Passagers/Coûts, Météo et Monitoring. Le croisement entre risque de disruption et niveau de congestion "
            "détermine le niveau d'alerte et les leviers proposés à l'OCC."
        )

    return (
        "Cet assistant aide l'OCC à comprendre le risque de disruption et la congestion associée, "
        "en expliquant les causes probables (congestion, météo, rotation, historique) et les actions recommandées. "
        "Pose une question sur les retards, une compagnie, la météo ou les actions opérationnelles pour plus de détails."
    )
