"""
backtest_ratings.py - Validación del motor de ratings.

Recorre las temporadas jornada a jornada. Para cada partido, ajusta los ratings
usando EXCLUSIVAMENTE los partidos anteriores a esa fecha y predice el
resultado. Ningún partido contribuye a su propia predicción.

Métricas:
  - log-loss : penaliza la sobreconfianza. Menor es mejor.
  - RPS      : Ranked Probability Score, tiene en cuenta que el empate está
               "entre" victoria local y visitante. Es la métrica estándar en
               predicción de fútbol. Menor es mejor.
  - Brier, acierto y calibración de empates como diagnóstico adicional.

Referencias para interpretar:
  - Predicción uniforme (1/3, 1/3, 1/3): log-loss 1.0986, RPS 0.2222
  - Un modelo publicable ronda 0.95-1.00 de log-loss y 0.19-0.21 de RPS
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ratings import fit_ratings, predict_match, apply_defaults
from src.utils import load_matches, logger

EPS = 1e-15


def rps(probs: np.ndarray, outcome: int) -> float:
    """
    Ranked Probability Score para 3 resultados ordenados (local, empate, visitante).
    outcome: 0 = local, 1 = empate, 2 = visitante
    """
    obs = np.zeros(3)
    obs[outcome] = 1.0
    cp, co = np.cumsum(probs), np.cumsum(obs)
    return float(np.sum((cp[:-1] - co[:-1]) ** 2) / 2.0)


def evaluate(
    matches: pd.DataFrame,
    start_season: str,
    half_life_days: float,
    prior_strength: float,
    fit_rho: bool,
    min_history: int = 500,
    use_defaults: bool = True,
) -> dict:
    """Backtest jornada a jornada desde `start_season`."""
    test = matches[matches["season"] >= start_season].copy()

    # Agrupamos por fecha: los ratings se reajustan una vez por jornada,
    # no una vez por partido. Mismo resultado, mucho más rápido.
    records = []
    for date, day in test.groupby("date"):
        history = matches[matches["date"] < date]
        if len(history) < min_history:
            continue

        r = fit_ratings(
            matches, as_of=date,
            half_life_days=half_life_days,
            prior_strength=prior_strength,
            fit_rho=fit_rho,
        )
        if use_defaults:
            teams_today = sorted(set(day["home"]) | set(day["away"]))
            r = apply_defaults(r, teams=teams_today)

        for _, m in day.iterrows():
            p = predict_match(r, m["home"], m["away"])
            hg, ag = m["home_goals"], m["away_goals"]
            outcome = 0 if hg > ag else (1 if hg == ag else 2)
            records.append({
                "date": date, "season": m["season"],
                "p_home": p["p_home"], "p_draw": p["p_draw"], "p_away": p["p_away"],
                "outcome": outcome,
            })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("El backtest no produjo predicciones")

    P = df[["p_home", "p_draw", "p_away"]].to_numpy()
    P = np.clip(P, EPS, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    y = df["outcome"].to_numpy()

    logloss = float(-np.mean(np.log(P[np.arange(len(y)), y])))
    rps_mean = float(np.mean([rps(P[i], y[i]) for i in range(len(y))]))

    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    brier = float(np.mean(np.sum((P - onehot) ** 2, axis=1)))

    acc = float(np.mean(P.argmax(axis=1) == y))

    return {
        "n": len(df),
        "logloss": logloss,
        "rps": rps_mean,
        "brier": brier,
        "accuracy": acc,
        "draws_pred": float(P[:, 1].mean()),
        "draws_real": float((y == 1).mean()),
        "detail": df,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-season", default="2015-16",
                   help="Primera temporada evaluada (default: 2015-16)")
    p.add_argument("--half-life", type=float, default=240.0)
    p.add_argument("--prior", type=float, default=6.0)
    p.add_argument("--sweep", action="store_true",
                   help="Probar varias vidas medias y comparar")
    args = p.parse_args()

    matches = load_matches()
    logger.info(f"Histórico: {len(matches):,} partidos, "
                f"{matches['season'].nunique()} temporadas")

    if args.sweep:
        print("\n  Calibrando vida media del decaimiento temporal")
        print("  " + "=" * 62)
        print(f"  {'vida media':>12} {'log-loss':>10} {'RPS':>9} {'acierto':>9} {'n':>7}")
        print("  " + "-" * 62)
        best = None
        for hl in [90, 120, 180, 240, 365, 540, 730, 1095]:
            res = evaluate(matches, args.start_season, hl, args.prior, True)
            flag = ""
            if best is None or res["rps"] < best[1]["rps"]:
                best = (hl, res)
                flag = "  <-"
            print(f"  {hl:>9.0f} d {res['logloss']:>10.4f} {res['rps']:>9.4f} "
                  f"{res['accuracy']*100:>8.1f}% {res['n']:>7}{flag}")
        print("  " + "=" * 62)
        print(f"  Mejor: {best[0]:.0f} días (RPS {best[1]['rps']:.4f})")
        return

    print("\n  Efecto de la corrección de Dixon-Coles")
    print("  " + "=" * 62)
    for label, use_rho in [("sin tau (Poisson puro)", False), ("con tau (Dixon-Coles)", True)]:
        t0 = time.time()
        res = evaluate(matches, args.start_season, args.half_life, args.prior, use_rho)
        print(f"\n  {label}")
        print(f"    partidos evaluados : {res['n']:,}")
        print(f"    log-loss           : {res['logloss']:.4f}")
        print(f"    RPS                : {res['rps']:.4f}")
        print(f"    Brier              : {res['brier']:.4f}")
        print(f"    acierto            : {res['accuracy']*100:.1f}%")
        print(f"    empates predichos  : {res['draws_pred']*100:.1f}%  "
              f"(reales {res['draws_real']*100:.1f}%)")
        print(f"    tiempo             : {time.time()-t0:.1f}s")

    print("\n  " + "-" * 62)
    print("  Referencia — predicción uniforme: log-loss 1.0986, RPS 0.2222")


if __name__ == "__main__":
    main()
