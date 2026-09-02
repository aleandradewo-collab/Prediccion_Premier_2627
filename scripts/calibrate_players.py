"""
calibrate_players.py - ¿Predice algo la cuota histórica de goles de un jugador?

Valida el reparto de goles de src/players.py con una pregunta directa y
verificable: tomando sólo información anterior al inicio de cada temporada
(2016/17-2025/26), ¿el jugador con más cuota histórica de goles de su equipo
es realmente el máximo goleador de ESE equipo esa temporada?

No es un log-loss/RPS como en los otros calibradores -aquí la pregunta es de
ranking (quién marca más), no de probabilidad de un resultado concreto- así
que la métrica es un simple acierto (hit-rate) por equipo-temporada, frente
a dos referencias ingenuas: elegir al azar entre la plantilla, o repetir el
máximo goleador de la temporada anterior.

Barre HALF_LIFE_DAYS para elegir la vida media que mejor acierta.

Uso:
    python scripts/calibrate_players.py
    python scripts/calibrate_players.py --start-season 2017-18
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.players import load_appearances, player_shares
from src.utils import load_matches, logger


def actual_top_scorers(appearances: pd.DataFrame, matches: pd.DataFrame, season: str,
                       team_ids: dict[str, int]) -> dict[str, str]:
    """Máximo goleador real de cada equipo en `season`, por player_club_id."""
    start = matches.loc[matches["season"] == season, "date"].min()
    end = matches.loc[matches["season"] == season, "date"].max() + pd.Timedelta(days=1)
    window = appearances[(appearances["date"] >= start) & (appearances["date"] < end)]

    out = {}
    for team, club_id in team_ids.items():
        sub = window[window["player_club_id"] == club_id]
        if sub.empty:
            continue
        totals = sub.groupby("player_name")["goals"].sum()
        if totals.max() > 0:
            out[team] = totals.idxmax()
    return out


def hit_rate(appearances: pd.DataFrame, matches: pd.DataFrame, start_season: str,
            half_life_days: float, team_ids: dict[str, int], baseline: str = "model") -> dict:
    seasons = sorted(s for s in matches["season"].unique() if s >= start_season)
    hits, n, prev_top = 0, 0, {}

    for season in seasons:
        as_of = matches.loc[matches["season"] == season, "date"].min()
        actual = actual_top_scorers(appearances, matches, season, team_ids)

        if baseline == "model":
            shares = player_shares(appearances, as_of=as_of, current_only=False,
                                   half_life_days=half_life_days)
            predicted = (shares.sort_values("share_goals", ascending=False)
                        .groupby("canonical")["player_name"].first().to_dict())
        elif baseline == "persistence":
            predicted = dict(prev_top)   # el goleador de la temporada anterior
        else:
            raise ValueError(baseline)

        for team, real in actual.items():
            pred = predicted.get(team)
            if pred is None:
                continue
            n += 1
            hits += int(pred == real)

        prev_top = actual

    return {"hits": hits, "n": n, "rate": hits / n if n else float("nan")}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-season", default="2016-17")
    p.add_argument("--half-lives", default="180,270,365,545,730")
    args = p.parse_args()

    matches = load_matches()
    logger.info(f"Histórico: {len(matches):,} partidos, {matches['season'].nunique()} temporadas")

    logger.info("Cargando apariciones de Premier League (appearances.csv)...")
    appearances = load_appearances()

    from src.squad import load_team_market_ids
    team_ids = dict(zip(load_team_market_ids()["canonical"], load_team_market_ids()["tm_club_id"]))

    print("\n  Vida media del peso de goles por jugador")
    print("  " + "=" * 50)
    print(f"  {'vida media':>12} {'acierto':>9} {'n':>6}")
    print("  " + "-" * 50)
    best = None
    for hl in [float(x) for x in args.half_lives.split(",")]:
        res = hit_rate(appearances, matches, args.start_season, hl, team_ids, baseline="model")
        flag = ""
        if best is None or res["rate"] > best[1]["rate"]:
            best, flag = (hl, res), "  <-"
        print(f"  {hl:>9.0f} d {res['rate']*100:>8.1f}% {res['n']:>6}{flag}")

    print("\n  Referencias")
    print("  " + "-" * 50)
    pers = hit_rate(appearances, matches, args.start_season, best[0], team_ids, baseline="persistence")
    n = pers["n"]
    naive = 1.0 / 20  # elegir al azar entre ~20 jugadores de plantilla con minutos relevantes
    print(f"  {'repetir goleador anterior':<26} {pers['rate']*100:>8.1f}% {n:>6}")
    print(f"  {'al azar (~20 candidatos)':<26} {naive*100:>8.1f}%")

    print("\n  " + "=" * 50)
    print(f"  Mejor vida media: {best[0]:.0f} días (acierto {best[1]['rate']*100:.1f}%)")
    print(f"  Copia a src/players.py:  HALF_LIFE_DAYS = {best[0]:.0f}")


if __name__ == "__main__":
    main()
