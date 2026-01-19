# src/weather_api.py
"""
Module météo pour STL (KSTL).

Utilise l'API Open-Meteo (gratuite, sans clé) pour récupérer
un snapshot météo et en déduire un indicateur de risque.
"""

from __future__ import annotations
import requests
from dataclasses import dataclass

# Coordonnées approx. de St. Louis Lambert (STL)
STL_LAT = 38.7487
STL_LON = -90.37


@dataclass
class WeatherSnapshot:
    temperature_c: float | None
    wind_speed_kt: float | None
    wind_direction_deg: float | None
    precipitation_mm: float | None
    cloud_cover_pct: float | None
    weather_code: int | None
    wx_risk_score_runtime: float


def _kmh_to_knots(v_kmh: float) -> float:
    return v_kmh * 0.539957


def _compute_runtime_risk(
    wind_speed_kt: float | None,
    precipitation_mm: float | None,
    cloud_cover_pct: float | None,
    weather_code: int | None,
) -> float:
    """
    Score de risque météo runtime [0,1] basé sur quelques heuristiques :
    - vent fort
    - précipitations
    - conditions météo "dures" (codes WMO)
    """
    score = 0.0

    # Vent
    if wind_speed_kt is not None:
        if wind_speed_kt <= 10:
            score += 0.1
        elif wind_speed_kt <= 20:
            score += 0.3
        else:
            score += 0.5

    # Précipitations
    if precipitation_mm is not None:
        if precipitation_mm > 0 and precipitation_mm <= 1:
            score += 0.1
        elif precipitation_mm > 1:
            score += 0.3

    # Nuages / visibilité indirecte
    if cloud_cover_pct is not None:
        if cloud_cover_pct > 80:
            score += 0.1

    # Codes météo "difficiles" (pluie forte, neige, orage)
    # D'après la table WMO Open‑Meteo (codes 61+, 71+, 80+, 95+).[web:44]
    if weather_code is not None:
        if weather_code >= 80 or weather_code in {61, 63, 65, 71, 73, 75, 95, 96, 99}:
            score += 0.3

    return float(max(0.0, min(1.0, score)))


def get_stl_weather_snapshot() -> WeatherSnapshot:
    """
    Récupère un snapshot météo pour STL via Open-Meteo.
    Retourne :
    - temperature_c : température actuelle à 2m (°C)
    - wind_speed_kt : vitesse du vent en noeuds
    - wind_direction_deg : direction du vent (deg)
    - precipitation_mm : précipitation instantanée (mm/h environ)
    - cloud_cover_pct : couverture nuageuse (%)
    - weather_code : code météo WMO
    - wx_risk_score_runtime : score [0,1] simple basé sur ces signaux

    En cas d'erreur API, retourne des valeurs par défaut raisonnables.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={STL_LAT}&longitude={STL_LON}"
        "&current=temperature_2m,wind_speed_10m,wind_direction_10m,"
        "precipitation,cloud_cover,weather_code"
        "&timezone=auto"
    )

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current", {})

        temp_c = float(current.get("temperature_2m")) if "temperature_2m" in current else None
        wind_speed_kmh = float(current.get("wind_speed_10m")) if "wind_speed_10m" in current else None
        wind_dir = float(current.get("wind_direction_10m")) if "wind_direction_10m" in current else None
        precip = float(current.get("precipitation")) if "precipitation" in current else None
        cloud = float(current.get("cloud_cover")) if "cloud_cover" in current else None
        wcode = int(current.get("weather_code")) if "weather_code" in current else None

        wind_speed_kt = _kmh_to_knots(wind_speed_kmh) if wind_speed_kmh is not None else None

        wx_risk = _compute_runtime_risk(
            wind_speed_kt=wind_speed_kt,
            precipitation_mm=precip,
            cloud_cover_pct=cloud,
            weather_code=wcode,
        )

        return WeatherSnapshot(
            temperature_c=temp_c,
            wind_speed_kt=wind_speed_kt,
            wind_direction_deg=wind_dir,
            precipitation_mm=precip,
            cloud_cover_pct=cloud,
            weather_code=wcode,
            wx_risk_score_runtime=wx_risk,
        )

    except Exception:
        # Fallback en cas d'erreur API
        return WeatherSnapshot(
            temperature_c=None,
            wind_speed_kt=8.0,            # vent modéré
            wind_direction_deg=None,
            precipitation_mm=0.0,
            cloud_cover_pct=None,
            weather_code=None,
            wx_risk_score_runtime=0.3,   # risque faible/modéré
        )
