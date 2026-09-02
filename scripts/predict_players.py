"""
predict_players.py - Predicciones individuales: Bota de Oro y máximo asistente.

Reparte los goles/asistencias que genera la simulación de equipos (Monte
Carlo, igual que simulate_season.py) entre los jugadores de cada plantilla,
según su cuota histórica de goles/asistencias (src/players.py). La
incertidumbre de "cuántos goles marca el equipo" viene del modelo de
equipo ya calibrado; lo único nuevo aquí es "de quién son esos goles".

Uso:
    python scripts/predict_players.py
    python scripts/predict_players.py --sims 50000 --top 30
    python scripts/predict_players.py --played results/jugados.csv
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.congestion import load_club_matches, team_date_index
from src.players import load_appearances, player_shares, simulate_player_stats
from src.ratings import apply_defaults, apply_squad_prior, fit_ratings
from src.simulator import simulate_season
from src.squad import current_squad_value
from src.utils import RESULTS_DIR, load_fixtures, load_matches, load_teams_2026_27, logger


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sims", type=int, default=20_000)
    p.add_argument("--as-of", default="2026-08-21")
    p.add_argument("--half-life", type=float, default=240.0)
    p.add_argument("--prior", type=float, default=4.0)
    p.add_argument("--played", type=Path,
                   help="CSV de partidos ya jugados: home, away, home_goals, away_goals")
    p.add_argument("--top", type=int, default=20, help="Filas a mostrar de cada tabla")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("\n" + "=" * 60)
    print("   PREMIER LEAGUE 2026/27 — BOTA DE ORO Y ASISTENCIAS")
    print("=" * 60 + "\n")
    t0 = time.time()

    matches = load_matches()
    fixtures = load_fixtures()
    teams = load_teams_2026_27()
    team_list = list(teams["canonical"])

    logger.info(f"Histórico: {len(matches):,} partidos")
    r = fit_ratings(matches, as_of=args.as_of, half_life_days=args.half_life,
                    prior_strength=args.prior, verbose=True)
    r = apply_defaults(r, teams=team_list, verbose=True)
    sv = current_squad_value(teams=team_list)
    r = apply_squad_prior(r, sv, verbose=True)
    date_index = team_date_index(load_club_matches())

    played = None
    if args.played:
        played = pd.read_csv(args.played)
        logger.info(f"Incorporando {len(played)} partidos ya jugados")

    res = simulate_season(r, fixtures, played=played, n_sims=args.sims,
                          seed=args.seed, date_index=date_index)
    logger.info(f"  {args.sims:,} temporadas simuladas")

    logger.info("Calculando cuota de goles/asistencias por jugador (appearances.csv)...")
    appearances = load_appearances()
    shares = player_shares(appearances, as_of=args.as_of, teams=team_list)
    logger.info(f"  {len(shares)} jugadores con historial en Premier")

    players = simulate_player_stats(res, shares, seed=args.seed)

    def show(df, col_p, col_media, titulo):
        print("\n" + "=" * 74)
        print(f"  {titulo}")
        print("=" * 74)
        print(f"  {'#':<3} {'Jugador':<22} {'Equipo':<16} {'Media':>7} {'P(1º)':>7}")
        print("  " + "-" * 70)
        for i, (_, row) in enumerate(df.sort_values(col_p, ascending=False).head(args.top).iterrows(), 1):
            print(f"  {i:<3} {row['player_name']:<22} {row['canonical']:<16} "
                  f"{row[col_media]:>7.2f} {row[col_p]*100:>6.1f}%")
        print("=" * 74)

    show(players, "p_bota_oro", "goles_medios", "BOTA DE ORO — máximo goleador de la liga")
    show(players, "p_max_asistente", "asist_medios", "MÁXIMO ASISTENTE")

    out = RESULTS_DIR / "player_predictions.csv"
    players.to_csv(out, index=False)
    logger.info(f"  player_predictions -> {out}")

    print(f"\n  Tiempo total: {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
