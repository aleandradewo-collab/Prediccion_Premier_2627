"""
congestion.py - Descanso entre partidos, en TODAS las competiciones de club.

Un equipo que juega la Champions League el martes llega al partido de
Premier del sábado con menos descanso que uno sin competición europea esa
semana. Esa fatiga sólo es visible si se cuentan TODOS los partidos de club
de `games.csv` (liga, FA Cup, Carabao, competiciones UEFA) y no sólo los de
Premier — que es lo único que ve `ratings.py` por defecto.

No cuenta partidos de selecciones (`competition_type ==
"national_team_competition"`): la fatiga que interesa aquí es la del
calendario de CLUB, no la de la ventana internacional.

LIMITACIÓN CONOCIDA: `games.csv` se corta en marzo de 2026, igual que
`player_valuations.csv` (ver README). Para cualquier fecha posterior no hay
partido previo registrado y `rest_days()` devuelve el valor por defecto — es
decir, para la temporada 2026/27 en sí este módulo no penaliza nada todavía;
empezará a hacerlo en cuanto se actualice `games.csv` con la temporada en
curso. Mientras tanto, su valor está en la validación histórica
(scripts/calibrate_congestion.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.squad import load_team_market_ids
from src.utils import DATA_RAW

NORMAL_REST_DAYS = 6.0   # descanso "normal": de una jornada de liga a la siguiente


def load_club_matches() -> pd.DataFrame:
    """
    Un partido de club por fila JUGADA (formato largo: una fila por cada
    equipo participante), con columnas canonical, date. Cubre todas las
    competiciones de club de games.csv, no sólo Premier.
    """
    g = pd.read_csv(DATA_RAW / "games.csv",
                    usecols=["date", "home_club_id", "away_club_id", "competition_type"])
    g = g[g["competition_type"] != "national_team_competition"]
    g["date"] = pd.to_datetime(g["date"])

    ids = load_team_market_ids().rename(columns={"tm_club_id": "club_id"})

    home = g[["date", "home_club_id"]].rename(columns={"home_club_id": "club_id"})
    away = g[["date", "away_club_id"]].rename(columns={"away_club_id": "club_id"})
    long = pd.concat([home, away], ignore_index=True)

    long = long.merge(ids, on="club_id")
    return long[["canonical", "date"]].sort_values("date").reset_index(drop=True)


def team_date_index(club_matches: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fechas de partido de cada equipo, ordenadas — para búsqueda O(log n)."""
    return {t: g["date"].to_numpy() for t, g in club_matches.groupby("canonical")}


def rest_days(date_index: dict[str, np.ndarray], team: str, date) -> float:
    """
    Días desde el último partido de `team` (cualquier competición)
    estrictamente antes de `date`.

    Sin partido previo registrado -ya sea porque el equipo arranca el
    histórico o porque `date` cae más allá de la cobertura de games.csv- se
    asume descanso normal: no hay información para penalizar, así que no se
    penaliza.
    """
    dates = date_index.get(team)
    if dates is None or len(dates) == 0:
        return NORMAL_REST_DAYS

    d = np.datetime64(pd.Timestamp(date))
    pos = np.searchsorted(dates, d)
    if pos == 0:
        return NORMAL_REST_DAYS
    return float((d - dates[pos - 1]) / np.timedelta64(1, "D"))


def matches_in_window(date_index: dict[str, np.ndarray], team: str, date,
                      window_days: float = 10.0) -> int:
    """
    Partidos de `team` (cualquier competición) en los `window_days` anteriores
    a `date`, sin contarlo a él mismo. Complementa a `rest_days`: un único
    hueco corto no es lo mismo que llevar tres partidos en diez días.
    """
    dates = date_index.get(team)
    if dates is None or len(dates) == 0:
        return 0

    d = np.datetime64(pd.Timestamp(date))
    start = d - np.timedelta64(int(window_days), "D")
    pos_end = np.searchsorted(dates, d)
    pos_start = np.searchsorted(dates, start)
    return int(pos_end - pos_start)
