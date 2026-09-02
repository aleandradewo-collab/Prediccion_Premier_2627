"""
calibrate_squad_prior.py - Mide el prior de valor de plantilla, no lo inventa.

Dos preguntas, en dos pasos:

  1. ¿Cómo se traduce el valor de plantilla en rating? Se ajusta un rating
     Dixon-Coles usando SÓLO los partidos de cada temporada (sin decaimiento
     entre temporadas) y se regresiona contra el valor de plantilla de esa
     temporada. La pendiente y el intercepto de esa regresión son
     SQUAD_ATTACK_SLOPE/INTERCEPT y SQUAD_DEFENSE_SLOPE/INTERCEPT en
     src/ratings.py.

  2. ¿Cuánto hay que mezclarlo con el rating histórico? Se barre el peso de
     la mezcla en el mismo backtest jornada a jornada de backtest_ratings.py
     y se elige el que minimiza el RPS. Ese peso es SQUAD_PRIOR_WEIGHT.

Uso:
    python scripts/calibrate_squad_prior.py                  # las dos preguntas
    python scripts/calibrate_squad_prior.py --start-season 2016-17
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ratings import (HALF_LIFE_DAYS, PRIOR_STRENGTH, apply_defaults,
                         apply_squad_prior, fit_ratings, predict_match,
                         set_squad_prior_coeffs)
from src.squad import squad_value_at
from src.utils import load_matches, load_teams_2026_27, logger

EPS = 1e-15
POST_WINDOW_DAYS = 45   # días tras el inicio de temporada: ventana de verano ya cerrada


def rps(probs: np.ndarray, outcome: int) -> float:
    obs = np.zeros(3)
    obs[outcome] = 1.0
    cp, co = np.cumsum(probs), np.cumsum(obs)
    return float(np.sum((cp[:-1] - co[:-1]) ** 2) / 2.0)


def season_squad_values(matches: pd.DataFrame, pv: pd.DataFrame) -> dict[str, pd.Series]:
    """Valor de plantilla por equipo, una vez por temporada, a +45 días del inicio."""
    out = {}
    for season, grp in matches.groupby("season"):
        start = grp["date"].min()
        teams = sorted(set(grp["home"]) | set(grp["away"]))
        out[season] = squad_value_at(start + pd.Timedelta(days=POST_WINDOW_DAYS),
                                     teams=teams, pv=pv)
    return out


# ── Paso 1: valor de plantilla -> rating ──────────────────────────────────────
def fit_value_to_rating(matches: pd.DataFrame, values: dict[str, pd.Series],
                        start_season: str) -> dict:
    """Regresión log(valor relativo) -> log(attack), log(defense) por equipo-temporada."""
    rows = []
    for season, grp in matches.groupby("season"):
        if season < start_season:
            continue
        sv = values[season].dropna()
        if len(sv) < 5:
            continue
        log_rel = np.log(sv / sv.mean())

        r = fit_ratings(grp, as_of=grp["date"].max() + pd.Timedelta(days=1),
                        half_life_days=100_000, prior_strength=PRIOR_STRENGTH)

        for team, lr in log_rel.items():
            if team not in r.attack:
                continue
            rows.append({"season": season, "team": team, "log_rel": lr,
                        "attack": r.attack[team], "defense": r.defense[team]})

    df = pd.DataFrame(rows)
    atk = linregress(df["log_rel"], np.log(df["attack"]))
    dfn = linregress(df["log_rel"], np.log(df["defense"]))
    return {"n": len(df), "attack": atk, "defense": dfn}


# ── Paso 2: barrido del peso de mezcla ────────────────────────────────────────
def evaluate_weight(matches: pd.DataFrame, values: dict[str, pd.Series],
                    weight: float, start_season: str, min_history: int = 500) -> dict:
    test = matches[matches["season"] >= start_season]

    records = []
    for date, day in test.groupby("date"):
        history = matches[matches["date"] < date]
        if len(history) < min_history:
            continue

        season = day["season"].iloc[0]
        teams_today = sorted(set(day["home"]) | set(day["away"]))

        r = fit_ratings(matches, as_of=date, half_life_days=HALF_LIFE_DAYS,
                        prior_strength=PRIOR_STRENGTH)
        r = apply_defaults(r, teams=teams_today)
        if weight > 0:
            r = apply_squad_prior(r, values[season], weight=weight)

        for _, m in day.iterrows():
            p = predict_match(r, m["home"], m["away"])
            hg, ag = m["home_goals"], m["away_goals"]
            outcome = 0 if hg > ag else (1 if hg == ag else 2)
            records.append({"p_home": p["p_home"], "p_draw": p["p_draw"],
                            "p_away": p["p_away"], "outcome": outcome})

    df = pd.DataFrame(records)
    P = np.clip(df[["p_home", "p_draw", "p_away"]].to_numpy(), EPS, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    y = df["outcome"].to_numpy()

    logloss = float(-np.mean(np.log(P[np.arange(len(y)), y])))
    rps_mean = float(np.mean([rps(P[i], y[i]) for i in range(len(y))]))
    return {"n": len(df), "logloss": logloss, "rps": rps_mean}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-season", default="2015-16")
    p.add_argument("--weights", default="0,0.05,0.1,0.15,0.2,0.25,0.3,0.4",
                   help="Pesos de mezcla a barrer, separados por coma")
    args = p.parse_args()

    matches = load_matches()
    pv = pd.read_csv("data/raw/player_valuations.csv")
    pv["date"] = pd.to_datetime(pv["date"])
    logger.info(f"Histórico: {len(matches):,} partidos, {matches['season'].nunique()} temporadas")

    logger.info("Calculando valor de plantilla por temporada (+45d del inicio)...")
    values = season_squad_values(matches, pv)

    print("\n  Paso 1 — valor de plantilla -> rating")
    print("  " + "=" * 62)
    fit = fit_value_to_rating(matches, values, args.start_season)
    atk, dfn = fit["attack"], fit["defense"]
    print(f"  {fit['n']} equipo-temporada evaluados")
    print(f"  attack  = exp({atk.intercept:+.4f} {atk.slope:+.4f} * log_rel)   "
          f"R²={atk.rvalue**2:.3f}")
    print(f"  defense = exp({dfn.intercept:+.4f} {dfn.slope:+.4f} * log_rel)   "
          f"R²={dfn.rvalue**2:.3f}")

    set_squad_prior_coeffs(atk.slope, atk.intercept, dfn.slope, dfn.intercept)

    print("\n  Paso 2 — barrido del peso de mezcla (backtest jornada a jornada)")
    print("  " + "=" * 62)
    print(f"  {'peso':>6} {'log-loss':>10} {'RPS':>9} {'n':>7}")
    print("  " + "-" * 62)
    best = None
    weights = [float(w) for w in args.weights.split(",")]
    for w in weights:
        res = evaluate_weight(matches, values, w, args.start_season)
        flag = ""
        if best is None or res["rps"] < best[1]["rps"]:
            best = (w, res)
            flag = "  <-"
        print(f"  {w:>6.2f} {res['logloss']:>10.4f} {res['rps']:>9.4f} {res['n']:>7}{flag}")

    print("  " + "=" * 62)
    print(f"  Mejor peso: {best[0]:.2f} (RPS {best[1]['rps']:.4f})")
    print("\n  Copia estos valores a src/ratings.py:")
    print(f"    SQUAD_ATTACK_SLOPE      = {atk.slope:.4f}")
    print(f"    SQUAD_ATTACK_INTERCEPT  = {atk.intercept:.4f}")
    print(f"    SQUAD_DEFENSE_SLOPE     = {dfn.slope:.4f}")
    print(f"    SQUAD_DEFENSE_INTERCEPT = {dfn.intercept:.4f}")
    print(f"    SQUAD_PRIOR_WEIGHT      = {best[0]:.2f}")


if __name__ == "__main__":
    main()
