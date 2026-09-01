"""
squad.py - Señal de valor de plantilla a partir de datos de Transfermarkt.

Los ratings de Dixon-Coles son puramente de resultados: no saben nada de los
fichajes de este verano. Este módulo añade una segunda fuente de información
-el valor de mercado agregado de la plantilla- que `ratings.apply_squad_prior()`
mezcla con el rating histórico.

Dos formas de medir el valor de plantilla, según para qué se usen:

1. HISTÓRICO (`squad_value_at`): usa player_valuations.csv, que trae una
   serie temporal de valoraciones por jugador junto con el club en el
   momento de la valoración. Sirve para calibrar el modelo sobre temporadas
   pasadas (scripts/calibrate_squad_prior.py).

2. ACTUAL (`current_squad_value`): player_valuations.csv no llega más allá
   de marzo de 2026, antes de que cerrara el mercado de verano (31 de
   agosto). Para la temporada 2026/27 se usa en su lugar el club y valor
   ACTUALES de players.csv, la única fuente que sí refleja los fichajes ya
   cerrados.
"""

from __future__ import annotations

import pandas as pd

from src.utils import DATA_RAW, COMP_PREMIER


def load_team_market_ids(path: pd.io.common.FilePath | None = None) -> pd.DataFrame:
    """canonical <-> tm_club_id, sólo equipos con dato en Transfermarkt.

    Coventry y otros ascendidos sin histórico no tienen tm_club_id: quedan
    fuera y `apply_squad_prior` los deja tal cual, igual que hace
    `apply_defaults` con los equipos sin rating.
    """
    tn = pd.read_csv(path or DATA_RAW / "team_names.csv")
    ids = tn.dropna(subset=["tm_club_id"])[["canonical", "tm_club_id"]].copy()
    ids["tm_club_id"] = ids["tm_club_id"].astype(int)
    return ids


def squad_value_at(as_of, teams: list[str] | None = None,
                   pv: pd.DataFrame | None = None) -> pd.Series:
    """
    Valor de plantilla por equipo, en euros, a partir de player_valuations.csv.

    Para cada jugador toma su última valoración anterior a `as_of` dentro de
    la Premier (GB1), en el club en el que jugaba entonces: así cuenta para
    el equipo que tenía su plantilla en esa fecha, no para el que tenga hoy.

    Args:
        as_of: fecha de corte
        teams: si se pasa, reindexa el resultado a esa lista (NaN si falta)
        pv:    player_valuations.csv ya cargado, para no releerlo en bucles
               que lo necesitan temporada a temporada (ver calibrate_squad_prior.py)
    """
    if pv is None:
        pv = pd.read_csv(DATA_RAW / "player_valuations.csv")
        pv["date"] = pd.to_datetime(pv["date"])

    as_of = pd.Timestamp(as_of)
    sub = pv[(pv["date"] < as_of) & (pv["player_club_domestic_competition_id"] == COMP_PREMIER)]
    if sub.empty:
        value = pd.Series(dtype=float)
    else:
        latest = sub.sort_values("date").groupby("player_id").tail(1)
        ids = load_team_market_ids()
        latest = latest.merge(ids, left_on="current_club_id", right_on="tm_club_id")
        value = latest.groupby("canonical")["market_value_in_eur"].sum()

    return value.reindex(teams) if teams is not None else value


def current_squad_value(teams: list[str] | None = None) -> pd.Series:
    """
    Valor de plantilla ACTUAL por equipo, a partir de players.csv.

    Único fichero del repositorio que refleja los fichajes ya cerrados del
    verano 2026: player_valuations.csv se corta en marzo, antes del cierre
    de mercado.
    """
    players = pd.read_csv(DATA_RAW / "players.csv")
    players = players.dropna(subset=["current_club_id", "market_value_in_eur"]).copy()
    players["current_club_id"] = players["current_club_id"].astype(int)

    ids = load_team_market_ids()
    merged = players.merge(ids, left_on="current_club_id", right_on="tm_club_id")
    value = merged.groupby("canonical")["market_value_in_eur"].sum()

    return value.reindex(teams) if teams is not None else value
