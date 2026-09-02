"""
season_trajectory.py - Cómo evolucionan las probabilidades de título/top-4/
descenso a lo largo de la temporada 2026/27, jornada a jornada.

Para cada jornada de --from a --to, simula la temporada usando como
`played` SÓLO los resultados reales ya conocidos hasta esa jornada -el
resto se simula- y guarda la tabla de probabilidades de ESE punto de la
temporada. Junta todas las jornadas en un único CSV para poder graficar
(en Excel, matplotlib, lo que sea) cómo cambia el título/top4/descenso de
cada equipo con cada resultado real que entra.

Los resultados "ya jugados" de cada jornada se detectan automáticamente
cruzando el calendario oficial (fixtures, con su columna matchday) contra
data/raw/epl_matches.csv de la temporada 2026/27 -no hace falta mantener un
CSV de jugados a mano-. Mientras esa temporada siga sin partidos cargados
en epl_matches.csv, todas las jornadas producen la misma proyección de
pretemporada: no es un bug, es que no hay resultado real que incorporar
todavía. En cuanto epl_matches.csv tenga la temporada en curso -manual o vía
scripts/update_current_season.py-, cada jornada usará lo que de verdad se
haya jugado hasta ahí.

Uso:
    python scripts/season_trajectory.py                # jornada 0 a la última con datos
    python scripts/season_trajectory.py --to 10          # sólo hasta la jornada 10
    python scripts/season_trajectory.py --sims 5000       # menos sims, barrido más rápido
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.congestion import load_club_matches, team_date_index
from src.ratings import apply_defaults, apply_squad_prior, fit_ratings
from src.simulator import simulate_season
from src.squad import current_squad_value
from src.utils import (RESULTS_DIR, load_fixtures, load_matches,
                       load_teams_2026_27, logger, matches_played_through)

SEASON_LABEL = "2026-27"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="from_md", type=int, default=0)
    p.add_argument("--to", dest="to_md", type=int, default=None,
                   help="Última jornada (por defecto, la última con algún resultado cargado)")
    p.add_argument("--sims", type=int, default=5_000,
                   help="Simulaciones por jornada (default 5000: 39 jornadas de golpe, prioriza velocidad)")
    p.add_argument("--as-of", default="2026-08-21", help="Fecha de corte para la jornada 0")
    p.add_argument("--half-life", type=float, default=240.0)
    p.add_argument("--prior", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    t0 = time.time()
    matches = load_matches()
    fixtures = load_fixtures()
    teams = load_teams_2026_27()
    team_list = list(teams["canonical"])

    season_matches = matches[matches["season"] == SEASON_LABEL].rename(
        columns={"home": "home", "away": "away"})
    logger.info(f"Partidos reales de {SEASON_LABEL} ya en epl_matches.csv: {len(season_matches)}")

    sv = current_squad_value(teams=team_list)
    date_index = team_date_index(load_club_matches())

    to_md = args.to_md
    if to_md is None:
        played_md = fixtures.merge(
            season_matches[["home", "away"]], on=["home", "away"], how="inner")["matchday"]
        to_md = int(played_md.max()) if len(played_md) else 0

    print(f"\n  Trayectoria jornada {args.from_md} a {to_md} · {args.sims:,} simulaciones cada una\n")

    rows = []
    for md in range(args.from_md, to_md + 1):
        played = matches_played_through(fixtures, season_matches, md)
        # as_of: el día después del último partido de esta jornada con fecha en el calendario,
        # o el as_of de pretemporada si aún no hay ninguna jornada con resultado real.
        md_dates = fixtures.loc[fixtures["matchday"] <= md, "date"]
        as_of = (md_dates.max() + pd.Timedelta(days=1)) if len(played) and len(md_dates) else args.as_of

        r = fit_ratings(matches, as_of=as_of, half_life_days=args.half_life, prior_strength=args.prior)
        r = apply_defaults(r, teams=team_list)
        r = apply_squad_prior(r, sv)

        res = simulate_season(r, fixtures, played=played if len(played) else None,
                              n_sims=args.sims, seed=args.seed, date_index=date_index)

        tabla = res.table.copy()
        tabla.insert(0, "matchday", md)
        tabla.insert(1, "partidos_jugados", len(played))
        rows.append(tabla)

        lider = tabla.sort_values("p_titulo", ascending=False).iloc[0]
        print(f"  J{md:<3} ({len(played):>3} partidos)  "
              f"líder: {lider['team']:<16} {lider['p_titulo']*100:>5.1f}% título")

    out = pd.concat(rows, ignore_index=True)
    csv_path = RESULTS_DIR / "title_trajectory.csv"
    out.to_csv(csv_path, index=False)

    xlsx_path = RESULTS_DIR / "title_trajectory.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        out.to_excel(xw, sheet_name="Trayectoria (detalle)", index=False)
        # Una hoja por métrica, ya pivotada jornada x equipo: lista para graficar
        # como líneas en Excel sin que el usuario tenga que armar la tabla dinámica.
        pivots = {
            "P(título)": "p_titulo", "P(top-4)": "p_top4",
            "P(descenso)": "p_descenso", "Puntos medios": "pts_medios",
        }
        for sheet, col in pivots.items():
            pivot = out.pivot(index="matchday", columns="team", values=col)
            pivot.to_excel(xw, sheet_name=sheet)

    print(f"\n  Guardado en {csv_path}")
    print(f"  Guardado en {xlsx_path} (con hojas por métrica ya pivotadas jornada x equipo)")
    print(f"  Tiempo total: {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
