"""
simulate_season.py - Predicción de la Premier League 2026/27.

Uso:
    python scripts/simulate_season.py                    # temporada completa
    python scripts/simulate_season.py --sims 50000
    python scripts/simulate_season.py --matchday 1       # sólo jornada 1
    python scripts/simulate_season.py --date 2026-08-22  # sólo esa fecha
    python scripts/simulate_season.py --played results/jugados.csv
    python scripts/simulate_season.py --save-raw         # + simulaciones crudas

Con --matchday o --date no se lanza Monte Carlo: se muestran las probabilidades
analíticas de esos partidos, que son exactas e instantáneas.

Con --played se incorporan los resultados ya disputados y sólo se simula lo que
queda de temporada. El CSV debe tener columnas: home, away, home_goals, away_goals.

Resultados en results/
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ratings import apply_defaults, fit_ratings
from src.simulator import export_results, predict_fixtures, simulate_season
from src.utils import (RESULTS_DIR, load_fixtures, load_matches,
                       load_teams_2026_27, logger)


def build_ratings(args):
    matches = load_matches()
    teams = load_teams_2026_27()
    logger.info(f"Histórico: {len(matches):,} partidos")

    r = fit_ratings(matches, as_of=args.as_of,
                    half_life_days=args.half_life,
                    prior_strength=args.prior, verbose=True)
    if not args.no_defaults:
        r = apply_defaults(r, teams=list(teams["canonical"]), verbose=args.verbose)
    return r


def show_matches(ratings, fixtures, titulo):
    """Modo partido a partido: probabilidades analíticas, sin Monte Carlo."""
    pred = predict_fixtures(ratings, fixtures)

    print("\n" + "=" * 74)
    print(f"  {titulo}")
    print("=" * 74)
    print(f"  {'Fecha':<12} {'Local':<16} {'Visitante':<16} "
          f"{'1':>6} {'X':>6} {'2':>6} {'Marc.':>7}")
    print("  " + "-" * 70)
    for _, m in pred.iterrows():
        fecha = pd.Timestamp(m["date"]).date() if pd.notna(m["date"]) else ""
        print(f"  {str(fecha):<12} {m['home']:<16} {m['away']:<16} "
              f"{m['p_home']*100:>5.1f}% {m['p_draw']*100:>5.1f}% "
              f"{m['p_away']*100:>5.1f}% {m['marcador_probable']:>7}")
    print("=" * 74)

    out = RESULTS_DIR / "match_predictions.csv"
    pred.to_csv(out, index=False)
    print(f"\n  Guardado en {out}\n")
    return pred


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sims", type=int, default=20_000,
                   help="Temporadas simuladas (default: 20000). Menos de 5000 es ruido")
    p.add_argument("--as-of", default="2026-08-21",
                   help="Fecha de corte de los ratings (default: inicio de temporada)")
    p.add_argument("--half-life", type=float, default=240.0)
    p.add_argument("--prior", type=float, default=4.0)
    p.add_argument("--matchday", type=int, help="Predecir sólo esta jornada (1-38)")
    p.add_argument("--date", help="Predecir sólo los partidos de esta fecha")
    p.add_argument("--played", type=Path,
                   help="CSV de partidos ya jugados: home, away, home_goals, away_goals")
    p.add_argument("--no-defaults", action="store_true",
                   help="No aplicar el prior de equipos ascendidos")
    p.add_argument("--save-raw", action="store_true",
                   help="Guardar también los puntos de cada simulación")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("\n" + "=" * 60)
    print("   PREMIER LEAGUE 2026/27 — PREDICTOR")
    print("=" * 60 + "\n")

    t0 = time.time()
    fixtures = load_fixtures()
    ratings = build_ratings(args)

    # ── Modo partido a partido ───────────────────────────────────────────────
    if args.matchday is not None:
        sub = fixtures[fixtures["matchday"] == args.matchday]
        if sub.empty:
            sys.exit(f"No hay partidos en la jornada {args.matchday}")
        show_matches(ratings, sub, f"JORNADA {args.matchday}")
        return

    if args.date is not None:
        sub = fixtures[fixtures["date"] == pd.Timestamp(args.date)]
        if sub.empty:
            sys.exit(f"No hay partidos el {args.date}")
        show_matches(ratings, sub, f"PARTIDOS DEL {args.date}")
        return

    # ── Modo temporada completa ──────────────────────────────────────────────
    played = None
    if args.played:
        played = pd.read_csv(args.played)
        req = {"home", "away", "home_goals", "away_goals"}
        if not req.issubset(played.columns):
            sys.exit(f"{args.played} debe tener las columnas {sorted(req)}")
        logger.info(f"Incorporando {len(played)} partidos ya jugados")

    res = simulate_season(ratings, fixtures, played=played,
                          n_sims=args.sims, seed=args.seed)

    print("\n" + "=" * 74)
    print(f"  CLASIFICACIÓN PROYECTADA — {args.sims:,} temporadas simuladas")
    if res.n_played_matches:
        print(f"  ({res.n_played_matches} partidos jugados, "
              f"{res.n_simulated_matches} simulados)")
    print("=" * 74)
    print(res.summary())
    print("=" * 74)

    campeon = res.table.iloc[0]
    print(f"\n  Favorito: {campeon['team']} "
          f"({campeon['p_titulo']*100:.1f}% de título, "
          f"{campeon['pts_medios']:.0f} puntos de media)")

    # Comprobaciones de coherencia: si alguna falla, hay un bug
    checks = [
        ("suma P(título)",   res.table["p_titulo"].sum(),   1.0),
        ("suma P(top-4)",    res.table["p_top4"].sum(),     4.0),
        ("suma P(descenso)", res.table["p_descenso"].sum(), 3.0),
    ]
    print("\n  Coherencia:")
    for nombre, valor, esperado in checks:
        ok = "OK" if abs(valor - esperado) < 1e-6 else "ERROR"
        print(f"    {nombre:<18} {valor:.4f}  (esperado {esperado:.1f})  [{ok}]")

    print("\n  Ficheros generados:")
    pred = predict_fixtures(ratings, fixtures)
    export_results(res, fixtures_pred=pred, save_raw=args.save_raw)

    print(f"\n  Tiempo total: {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
