"""
players.py - Predicciones individuales (Bota de Oro, asistencias).

No es un modelo de goles por jugador independiente del motor de equipos: eso
exigiría predecir minutos, titularidades y rotación, para lo que no hay datos
de convocatorias futuras. En su lugar, REPARTE lo que el simulador de
equipos ya genera.

`simulate_season()` produce, para cada una de las n_sims temporadas
simuladas, los goles totales de cada equipo (`res.goals_for`). Este módulo
estima, para cada jugador de la plantilla actual, qué CUOTA de los goles de
su equipo suele anotar -medida sobre sus apariciones reales en Premier,
ponderadas por recencia- y reparte los goles de cada simulación entre los
jugadores según esa cuota mediante un sorteo multinomial. Así el reparto
hereda toda la incertidumbre y calibración del modelo de equipo: un equipo
que en una simulación marca muchos goles reparte más goles entre sus
jugadores en ESA simulación, no una media fija.

Las asistencias se derivan igual, escalando los goles de equipo por
ASSISTS_PER_GOAL (medido sobre el propio dataset, ver calibrate_players.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.squad import load_team_market_ids
from src.utils import COMP_PREMIER, DATA_RAW

# Medido en scripts/calibrate_players.py: barrido de vida media 180-730 días
# sobre el acierto de "quién es el máximo goleador del equipo esta temporada",
# 2016/17-2025/26. Óptimo en 270 días, pero la diferencia con el resto del
# rango es pequeña (33.7%-36.3%) — no es un pico afilado como el de
# HALF_LIFE_DAYS en ratings.py. Ver README para la comparación con las
# referencias ingenuas.
HALF_LIFE_DAYS = 270.0
ASSISTS_PER_GOAL = 0.792  # medido: asistencias / goles en GB1, 2012/13-2025/26


def load_appearances(competition: str = COMP_PREMIER) -> pd.DataFrame:
    """Apariciones de `competition` (GB1 = Premier League por defecto)."""
    app = pd.read_csv(DATA_RAW / "appearances.csv",
                      usecols=["player_id", "player_club_id", "player_current_club_id",
                              "date", "player_name", "competition_id",
                              "goals", "assists", "minutes_played"])
    app = app[app["competition_id"] == competition].copy()
    app["date"] = pd.to_datetime(app["date"])
    return app


def player_shares(appearances: pd.DataFrame, as_of, teams: list[str] | None = None,
                  half_life_days: float = HALF_LIFE_DAYS, current_only: bool = True) -> pd.DataFrame:
    """
    Cuota de cada jugador sobre los goles y asistencias de su equipo,
    ponderada por recencia.

    `current_only=True` (predicción real, uso por defecto): sólo cuentan las
    apariciones jugadas PARA el club ACTUAL del jugador (`player_club_id ==
    player_current_club_id`) — los goles marcados en un club anterior no
    dicen nada sobre su aportación en el nuevo. Misma lógica que
    `current_squad_value()` en squad.py, aplicada partido a partido en vez de
    a un valor de mercado.

    `current_only=False` (backtest histórico, ver calibrate_players.py):
    `player_current_club_id` sólo refleja el club MÁS RECIENTE del jugador en
    el snapshot de Transfermarkt, no cuál era "su club actual" en una fecha
    pasada — no sirve para reconstruir la plantilla de una temporada
    anterior. Se agrupa directamente por `player_club_id` (el club en el que
    jugó cada partido) y es la propia ponderación por recencia la que hace
    que un jugador que se fue deje de pesar.

    Devuelve una fila por jugador con columnas: player_id, player_name,
    canonical (equipo), goals_w, assists_w (goles/asistencias ponderados por
    recencia), share_goals, share_assists (normalizados para sumar 1 dentro
    de cada equipo), minutes_w.
    """
    as_of = pd.Timestamp(as_of)
    df = appearances[appearances["date"] < as_of].copy()
    if current_only:
        df = df[df["player_club_id"] == df["player_current_club_id"]]

    days = (as_of - df["date"]).dt.total_seconds() / 86400.0
    df["w"] = np.exp(-np.log(2.0) * np.maximum(days, 0.0) / half_life_days)

    ids = load_team_market_ids().rename(columns={"tm_club_id": "player_club_id"})
    df = df.merge(ids, on="player_club_id")

    g = (df.groupby(["player_id", "player_name", "canonical"])
        .apply(lambda d: pd.Series({
            "goals_w":   float((d["w"] * d["goals"]).sum()),
            "assists_w": float((d["w"] * d["assists"]).sum()),
            "minutes_w": float((d["w"] * d["minutes_played"]).sum()),
        }), include_groups=False)
        .reset_index())

    totals = g.groupby("canonical")[["goals_w", "assists_w"]].transform("sum")
    g["share_goals"] = np.where(totals["goals_w"] > 0, g["goals_w"] / totals["goals_w"], 0.0)
    g["share_assists"] = np.where(totals["assists_w"] > 0, g["assists_w"] / totals["assists_w"], 0.0)

    if teams is not None:
        g = g[g["canonical"].isin(teams)]
    return g.sort_values(["canonical", "share_goals"], ascending=[True, False]).reset_index(drop=True)


def simulate_player_stats(res, shares: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
    """
    Reparte los goles/asistencias de cada simulación de equipo (`res.goals_for`,
    de `simulate_season()`) entre los jugadores de su plantilla, en proporción
    a `share_goals`/`share_assists`.

    Cada simulación de equipo se reparte con un sorteo multinomial -no una
    cuota fija-, así que un equipo que en una simulación concreta marca más
    de lo habitual también reparte más goles entre sus jugadores EN ESA
    simulación: la incertidumbre del modelo de equipo se propaga a los
    jugadores en vez de perderse.

    Equipos sin plantilla en `shares` (p.ej. Coventry, sin dato de
    Transfermarkt) quedan fuera del resultado, no reciben goles.

    LIMITACIÓN: la cuota de cada jugador (`share_goals`/`share_assists`) es
    FIJA en las n_sims simulaciones — sólo varía cuántos goles marca el
    equipo, no quién se los reparte dentro de él. En la realidad esa cuota
    también fluctúa (lesiones, cambios tácticos, un fichaje de enero), así
    que las probabilidades de Bota de Oro están algo más ajustadas de lo que
    debería ser una predicción con toda la incertidumbre real incorporada.
    """
    rng = np.random.default_rng(seed)
    team_idx = {t: i for i, t in enumerate(res.teams)}

    goals_blocks, assists_blocks, rows = [], [], []
    for team, grp in shares.groupby("canonical", sort=False):
        if team not in team_idx or grp.empty:
            continue
        team_goals = np.round(res.goals_for[:, team_idx[team]]).astype(int)
        team_assists = np.round(team_goals * ASSISTS_PER_GOAL).astype(int)

        pvals_g = grp["share_goals"].to_numpy()
        pvals_a = grp["share_assists"].to_numpy()
        pvals_g = pvals_g / pvals_g.sum() if pvals_g.sum() > 0 else np.ones(len(grp)) / len(grp)
        pvals_a = pvals_a / pvals_a.sum() if pvals_a.sum() > 0 else np.ones(len(grp)) / len(grp)

        goals_blocks.append(rng.multinomial(team_goals, pvals_g))
        assists_blocks.append(rng.multinomial(team_assists, pvals_a))
        rows.append(grp[["player_id", "player_name", "canonical"]].reset_index(drop=True))

    goals = np.concatenate(goals_blocks, axis=1)
    assists = np.concatenate(assists_blocks, axis=1)
    players = pd.concat(rows, ignore_index=True)

    top_scorer = np.bincount(goals.argmax(axis=1), minlength=goals.shape[1])
    top_assist = np.bincount(assists.argmax(axis=1), minlength=assists.shape[1])

    out = players.copy()
    out["goles_medios"] = goals.mean(axis=0)
    out["goles_p10"] = np.percentile(goals, 10, axis=0)
    out["goles_p90"] = np.percentile(goals, 90, axis=0)
    out["p_bota_oro"] = top_scorer / res.n_sims
    out["asist_medios"] = assists.mean(axis=0)
    out["p_max_asistente"] = top_assist / res.n_sims

    return out.sort_values("p_bota_oro", ascending=False).reset_index(drop=True)
