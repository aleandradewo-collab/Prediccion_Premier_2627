"""
calibrate_congestion.py - Mide el efecto de la fatiga de calendario, no lo asume.

Un único pase de backtest jornada a jornada (rating SIN fatiga + goles reales)
se reutiliza para probar DOS formas de medir la congestión de calendario,
porque una sola podría fallar por mala especificación y no porque el efecto
no exista:

  A. DÉFICIT DE DESCANSO: días desde el último partido de club (cualquier
     competición), frente a NORMAL_REST_DAYS. Corto = fatiga puntual.
  B. CARGA RECIENTE: partidos jugados en los últimos 10 días. Distingue a un
     equipo con calendario apretado de uno con un único hueco corto.

Para cada una: ajuste por máxima verosimilitud de una regresión de Poisson
con offset (mismo enfoque que _fit_rho en src/ratings.py) y comparación de
log-loss/RPS con y sin el ajuste — igual que la tabla "sin tau / con tau" de
backtest_ratings.py. Si ninguna mejora el backtest, la conclusión honesta es
que este dataset no muestra un efecto de fatiga detectable, y así se reporta.

El descanso se calcula con TODAS las competiciones de club (games.csv), no
sólo Premier: ver src/congestion.py.

Uso:
    python scripts/calibrate_congestion.py
    python scripts/calibrate_congestion.py --start-season 2016-17 --window 14
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.congestion import (NORMAL_REST_DAYS, load_club_matches, matches_in_window,
                            rest_days, team_date_index)
from src.ratings import (HALF_LIFE_DAYS, PRIOR_STRENGTH, apply_defaults,
                         fit_ratings, predict_lambdas, score_matrix)
from src.utils import load_matches, logger

EPS = 1e-15
NORMAL_LOAD = 2.0   # partidos "normales" en 10 días para un equipo sin Europa entre semana


def rps(probs: np.ndarray, outcome: int) -> float:
    obs = np.zeros(3)
    obs[outcome] = 1.0
    cp, co = np.cumsum(probs), np.cumsum(obs)
    return float(np.sum((cp[:-1] - co[:-1]) ** 2) / 2.0)


def collect_records(matches: pd.DataFrame, start_season: str,
                    min_history: int = 500) -> pd.DataFrame:
    """
    Un pase de backtest jornada a jornada: rating SIN fatiga + resultado real.
    Sin ninguna métrica de congestión todavía — se calculan aparte a partir de
    date/home/away, para poder probar varias sin repetir el ajuste de ratings.
    """
    test = matches[matches["season"] >= start_season]
    rows = []

    for date, day in test.groupby("date"):
        history = matches[matches["date"] < date]
        if len(history) < min_history:
            continue

        teams_today = sorted(set(day["home"]) | set(day["away"]))
        r = fit_ratings(matches, as_of=date, half_life_days=HALF_LIFE_DAYS,
                        prior_strength=PRIOR_STRENGTH)
        r = apply_defaults(r, teams=teams_today)

        for _, m in day.iterrows():
            home, away = m["home"], m["away"]
            lh, la = predict_lambdas(r, home, away)
            hg, ag = int(m["home_goals"]), int(m["away_goals"])
            outcome = 0 if hg > ag else (1 if hg == ag else 2)
            rows.append({"date": date, "home": home, "away": away,
                        "lh": lh, "la": la, "hg": hg, "ag": ag, "outcome": outcome})

    return pd.DataFrame(rows)


def add_congestion_features(df: pd.DataFrame, date_index: dict, window_days: float) -> pd.DataFrame:
    df = df.copy()
    df["rest_home"] = [rest_days(date_index, h, d) for h, d in zip(df["home"], df["date"])]
    df["rest_away"] = [rest_days(date_index, a, d) for a, d in zip(df["away"], df["date"])]
    df["load_home"] = [matches_in_window(date_index, h, d, window_days)
                       for h, d in zip(df["home"], df["date"])]
    df["load_away"] = [matches_in_window(date_index, a, d, window_days)
                       for a, d in zip(df["away"], df["date"])]
    return df


def fit_coef(df: pd.DataFrame, covariate_home: np.ndarray, covariate_away: np.ndarray) -> float:
    """MLE de un coeficiente sobre una regresión de Poisson con offset:
    goles ~ Poisson(lambda_base * exp(coef * covariate))."""
    lam = np.concatenate([df["lh"], df["la"]])
    cov = np.concatenate([covariate_home, covariate_away])
    goals = np.concatenate([df["hg"], df["ag"]])

    def neg_ll(coef):
        mu = lam * np.exp(coef * cov)
        return -float(np.sum(goals * np.log(mu) - mu))

    res = minimize_scalar(neg_ll, bounds=(-0.5, 0.5), method="bounded")
    return float(res.x)


def evaluate(df: pd.DataFrame, cov_home: np.ndarray, cov_away: np.ndarray, coef: float) -> dict:
    """log-loss/RPS de las probabilidades 1X2 aplicando exp(-coef * covariate)."""
    lh = df["lh"].to_numpy() * np.exp(-coef * cov_home)
    la = df["la"].to_numpy() * np.exp(-coef * cov_away)
    y = df["outcome"].to_numpy()

    P = np.empty((len(df), 3))
    for i in range(len(df)):
        m = score_matrix(lh[i], la[i], rho=0.0)
        P[i] = [np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()]

    P = np.clip(P, EPS, 1.0)
    P = P / P.sum(axis=1, keepdims=True)

    logloss = float(-np.mean(np.log(P[np.arange(len(y)), y])))
    rps_mean = float(np.mean([rps(P[i], y[i]) for i in range(len(y))]))
    return {"n": len(y), "logloss": logloss, "rps": rps_mean}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-season", default="2015-16")
    p.add_argument("--window", type=float, default=10.0,
                   help="Días de la ventana de carga reciente (opción B)")
    args = p.parse_args()

    matches = load_matches()
    logger.info(f"Histórico: {len(matches):,} partidos, {matches['season'].nunique()} temporadas")

    logger.info("Cargando calendario de todas las competiciones (games.csv)...")
    date_index = team_date_index(load_club_matches())

    logger.info("Backtest jornada a jornada (rating sin fatiga)...")
    df = collect_records(matches, args.start_season)
    df = add_congestion_features(df, date_index, args.window)
    print(f"\n  {len(df):,} partidos evaluados")

    base = evaluate(df, np.zeros(len(df)), np.zeros(len(df)), 0.0)

    print("\n  A. Déficit de descanso (días bajo NORMAL_REST_DAYS)")
    print("  " + "=" * 62)
    deficit_h = np.maximum(0.0, NORMAL_REST_DAYS - df["rest_home"].to_numpy())
    deficit_a = np.maximum(0.0, NORMAL_REST_DAYS - df["rest_away"].to_numpy())
    coef_a = fit_coef(df, deficit_h, deficit_a)
    res_a = evaluate(df, deficit_h, deficit_a, coef_a)
    print(f"  descanso corto en el {100*(deficit_h>0).mean():.1f}% de los locales")
    print(f"  coef (MLE) = {coef_a:+.4f}   log-loss {res_a['logloss']:.4f}   RPS {res_a['rps']:.4f}")

    print("\n  B. Carga reciente (partidos en los últimos "
          f"{args.window:.0f} días, sobre NORMAL_LOAD={NORMAL_LOAD:.0f})")
    print("  " + "=" * 62)
    over_h = np.maximum(0.0, df["load_home"].to_numpy() - NORMAL_LOAD)
    over_a = np.maximum(0.0, df["load_away"].to_numpy() - NORMAL_LOAD)
    coef_b = fit_coef(df, over_h, over_a)
    res_b = evaluate(df, over_h, over_a, coef_b)
    print(f"  carga por encima de lo normal en el {100*(over_h>0).mean():.1f}% de los locales")
    print(f"  coef (MLE) = {coef_b:+.4f}   log-loss {res_b['logloss']:.4f}   RPS {res_b['rps']:.4f}")

    print("\n  Referencia — sin ningún ajuste de fatiga")
    print("  " + "=" * 62)
    print(f"  log-loss {base['logloss']:.4f}   RPS {base['rps']:.4f}")

    best = min([("A", coef_a, res_a), ("B", coef_b, res_b)], key=lambda x: x[2]["rps"])
    print("\n  " + "-" * 62)
    if best[2]["rps"] < base["rps"] - 1e-4:
        print(f"  Opción {best[0]} mejora el backtest de forma perceptible. "
              f"Copia a src/ratings.py:")
        print(f"    CONGESTION_COEF = {best[1]:.4f}")
    else:
        print("  Ninguna de las dos formas de medir la congestión mejora el backtest de")
        print("  forma perceptible (diferencia de RPS por debajo de 0.0001). Conclusión")
        print("  honesta: este dataset no muestra un efecto de fatiga detectable a nivel")
        print("  de Premier League. CONGESTION_COEF se deja en 0 — el mecanismo queda")
        print("  implementado y listo para recalibrar si esto cambia con más temporadas.")


if __name__ == "__main__":
    main()
