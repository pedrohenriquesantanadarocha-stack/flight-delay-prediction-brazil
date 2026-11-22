"""Portuguese display labels for figures and LaTeX tables of the paper."""

from __future__ import annotations

MODEL_NAMES = {
    "dummy": "Baseline (prevalência)",
    "logistic_regression": "Regressão Logística",
    "decision_tree": "Árvore de Decisão",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "hist_gradient_boosting": "HistGradientBoosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

GROUP_LABELS = {
    "calendar": "Calendário",
    "traffic": "Tráfego",
    "airline_route": "Companhia e rota",
    "weather_origin": "Clima na origem",
    "weather_destination": "Clima no destino",
}

_WEATHER = {
    "precip_mm": "Precipitação (mm)",
    "wind_speed_kmh": "Vento médio (km/h)",
    "wind_gust_kmh": "Rajada de vento (km/h)",
    "dewpoint_depression_c": "Depressão do ponto de orvalho (°C)",
    "temp_range_c": "Amplitude térmica (°C)",
    "temp_mean_c": "Temperatura média (°C)",
}

FEATURE_LABELS = {
    "dep_hour": "Hora da partida",
    "day_of_week": "Dia da semana",
    "high_season": "Alta temporada",
    "weekend": "Fim de semana",
    "hour_sin": "Hora (seno)", "hour_cos": "Hora (cosseno)",
    "dow_sin": "Dia da semana (seno)", "dow_cos": "Dia da semana (cosseno)",
    "month_sin": "Mês (seno)", "month_cos": "Mês (cosseno)",
    "origin_hourly_flow": "Voos programados na hora (origem)",
    "scheduled_duration_min": "Duração prevista (min)",
    "airline_icao": "Companhia aérea",
    "origin_icao": "Aeroporto de origem",
    "destination_icao": "Aeroporto de destino",
    "route": "Rota",
    **{f"origin_{k}": f"{v}, origem" for k, v in _WEATHER.items()},
    **{f"destination_{k}": f"{v}, destino" for k, v in _WEATHER.items()},
}

OPERATING_POINTS = {
    "default_0.5": "Padrão (0,5)",
    "f_beta": "F-beta máximo",
    "recall_target": "Recall-alvo",
    "cost": "Custo mínimo",
}


def label(name: str) -> str:
    """Portuguese label for a feature, group or model; falls back to the name itself."""
    for table in (FEATURE_LABELS, GROUP_LABELS, MODEL_NAMES, OPERATING_POINTS):
        if name in table:
            return table[name]
    return name
