"""
simulator.py - Simulación Monte Carlo de una temporada completa de liga.

Tres niveles de granularidad:

  1. predict_fixtures()  -> un partido por fila: lambdas, 1X2, marcador más
                            probable. No usa Monte Carlo, es analítico.
  2. simulate_season()   -> N temporadas completas agregadas en probabilidades
                            de título, top-4 y descenso.
  3. export_results()    -> vuelca todo a CSV, incluida la distribución de
                            puntos por equipo.

La salida NO es "gana el City" sino "City 34%, Arsenal 27%...". Una liga de 38
jornadas tiene muchísimo ruido: con una sola simulación el resultado es
prácticamente aleatorio, por eso el mínimo utilizable son varios miles.

IMPLEMENTACIÓN VECTORIZADA
--------------------------
En lugar de recorrer partido a partido en Python, se muestrean de golpe todas
las matrices de goles (n_sims × n_partidos) y se acumulan por equipo mediante
producto matricial con matrices de incidencia local/visitante.

DESEMPATES
----------
La Premier ordena por: puntos, diferencia de goles, goles a favor. Si persiste
el empate en puestos decisivos se juega un partido de desempate, algo que aquí
se resuelve aleatoriamente: ocurre en una fracción ínfima de simulaciones y no
altera las probabilidades agregadas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ratings import Ratings, match_probabilities, predict_lambdas, score_matrix
from src.utils import RESULTS_DIR, logger

CHAMPIONS_SPOTS  = 4    # top 4 -> Champions League
RELEGATION_SPOTS = 3    # últimos 3 -> descienden


# ── 1. Predicción partido a partido (analítica, sin Monte Carlo) ─────────────
def predict_fixtures(ratings: Ratings, fixtures: pd.DataFrame,
                     max_goals: int = 10) -> pd.DataFrame:
    """
    Probabilidades de cada partido del calendario, una fila por partido.

    No simula: calcula la matriz de marcadores analíticamente. Es instantáneo y
    exacto, así que sirve tanto para los 380 partidos de la temporada como para
    una sola jornada.

    Columnas: date, matchday, home, away, lambda_home, lambda_away,
              p_home, p_draw, p_away, xpts_home, xpts_away,
              marcador más probable y su probabilidad.
    """
    rows = []
    for _, m in fixtures.iterrows():
        lh, la = predict_lambdas(ratings, m["home"], m["away"])
        p = match_probabilities(lh, la, ratings.rho, max_goals)

        mat = score_matrix(lh, la, ratings.rho, max_goals)
        hg, ag = np.unravel_index(mat.argmax(), mat.shape)

        rows.append({
            "date":        m.get("date"),
            "matchday":    m.get("matchday"),
            "home":        m["home"],
            "away":        m["away"],
            "lambda_home": round(lh, 3),
            "lambda_away": round(la, 3),
            "p_home":      round(p["p_home"], 4),
            "p_draw":      round(p["p_draw"], 4),
            "p_away":      round(p["p_away"], 4),
            "xpts_home":   round(p["xpts_home"], 3),
            "xpts_away":   round(p["xpts_away"], 3),
            "marcador_probable": f"{hg}-{ag}",
            "p_marcador":  round(float(mat[hg, ag]), 4),
        })

    return pd.DataFrame(rows)


# ── 2. Simulación Monte Carlo de la temporada ────────────────────────────────
@dataclass
class SeasonResults:
    """Resultado agregado de N temporadas simuladas."""
    table:     pd.DataFrame   # una fila por equipo con las probabilidades
    points:    np.ndarray     # (n_sims, n_teams) puntos finales
    positions: np.ndarray     # (n_sims, n_teams) posición final
    goals_for: np.ndarray     # (n_sims, n_teams)
    goals_against: np.ndarray
    teams:     list[str]
    n_sims:    int
    n_simulated_matches: int
    n_played_matches: int = 0

    def summary(self, top: int | None = None) -> str:
        df = self.table if top is None else self.table.head(top)
        lines = [
            f"  {'#':<3} {'Equipo':<16} {'Pts':>6} {'P10-P90':>10} "
            f"{'Título':>8} {'Top-4':>8} {'Desc.':>8}",
            "  " + "-" * 68,
        ]
        for i, (_, r) in enumerate(df.iterrows(), start=1):
            rango = f"{r['pts_p10']:.0f}-{r['pts_p90']:.0f}"
            lines.append(
                f"  {i:<3} {r['team']:<16} {r['pts_medios']:>6.1f} {rango:>10} "
                f"{r['p_titulo']*100:>7.1f}% {r['p_top4']*100:>7.1f}% "
                f"{r['p_descenso']*100:>7.1f}%"
            )
        return "\n".join(lines)

    def points_distribution(self, bins: int = 20) -> pd.DataFrame:
        """Histograma de puntos finales por equipo."""
        out = []
        for i, t in enumerate(self.teams):
            counts, edges = np.histogram(self.points[:, i], bins=bins)
            for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
                out.append({"team": t, "pts_min": round(lo, 1),
                            "pts_max": round(hi, 1), "prob": c / self.n_sims})
        return pd.DataFrame(out)

    def position_matrix(self) -> pd.DataFrame:
        """P(cada equipo termine en cada puesto). Filas equipo, columnas puesto."""
        n = len(self.teams)
        m = np.zeros((n, n))
        for i in range(n):
            for pos in range(1, n + 1):
                m[i, pos - 1] = (self.positions[:, i] == pos).mean()
        df = pd.DataFrame(m, index=self.teams, columns=range(1, n + 1))
        return df.loc[self.table["team"]]


def simulate_season(
    ratings: Ratings,
    fixtures: pd.DataFrame,
    played: pd.DataFrame | None = None,
    n_sims: int = 20_000,
    seed: int | None = 42,
    max_goals: int = 12,
) -> SeasonResults:
    """
    Simula la temporada n_sims veces.

    Args:
        ratings:  ratings ya ajustados (pasa antes por apply_defaults)
        fixtures: calendario completo con columnas home, away
        played:   partidos ya disputados (home, away, home_goals, away_goals).
                  Si se pasa, esos partidos NO se simulan: sus puntos y goles se
                  suman como base y sólo se simula lo que queda. Es lo que
                  permite recalcular probabilidades cada jornada durante la
                  temporada.
        n_sims:   temporadas simuladas. Menos de 5.000 es ruido puro
    """
    teams = sorted(set(fixtures["home"]) | set(fixtures["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    # ── Separar jugados de pendientes ────────────────────────────────────────
    base_pts = np.zeros(n_teams)
    base_gf  = np.zeros(n_teams)
    base_ga  = np.zeros(n_teams)
    n_played = 0

    pend = fixtures
    if played is not None and len(played):
        ya = set(zip(played["home"], played["away"]))
        mask = [(h, a) not in ya for h, a in zip(fixtures["home"], fixtures["away"])]
        pend = fixtures[mask]

        for _, m in played.iterrows():
            ih, ia = idx.get(m["home"]), idx.get(m["away"])
            if ih is None or ia is None:
                continue
            hg, ag = int(m["home_goals"]), int(m["away_goals"])
            base_pts[ih] += 3 if hg > ag else (1 if hg == ag else 0)
            base_pts[ia] += 3 if ag > hg else (1 if hg == ag else 0)
            base_gf[ih] += hg; base_ga[ih] += ag
            base_gf[ia] += ag; base_ga[ia] += hg
            n_played += 1

    n_fix = len(pend)
    if n_fix == 0:
        raise ValueError("No queda ningún partido por simular")

    h = pend["home"].map(idx).to_numpy()
    a = pend["away"].map(idx).to_numpy()

    lam_h = np.empty(n_fix)
    lam_a = np.empty(n_fix)
    for i, (_, m) in enumerate(pend.iterrows()):
        lam_h[i], lam_a[i] = predict_lambdas(ratings, m["home"], m["away"])

    logger.info(f"  simulando {n_sims:,} temporadas · {n_fix} partidos pendientes"
                + (f" · {n_played} ya jugados" if n_played else ""))

    rng = np.random.default_rng(seed)
    hg = rng.poisson(lam_h, size=(n_sims, n_fix)).clip(0, max_goals)
    ag = rng.poisson(lam_a, size=(n_sims, n_fix)).clip(0, max_goals)

    draw  = hg == ag
    pts_h = np.where(hg > ag, 3, np.where(draw, 1, 0))
    pts_a = np.where(hg < ag, 3, np.where(draw, 1, 0))

    H = np.zeros((n_fix, n_teams)); H[np.arange(n_fix), h] = 1.0
    A = np.zeros((n_fix, n_teams)); A[np.arange(n_fix), a] = 1.0

    points = pts_h @ H + pts_a @ A + base_pts[None, :]
    gf     = hg @ H + ag @ A + base_gf[None, :]
    ga     = ag @ H + hg @ A + base_ga[None, :]
    gd     = gf - ga

    positions = _rank(points, gd, gf, rng)

    table = pd.DataFrame({
        "team":       teams,
        "pts_medios": points.mean(axis=0),
        "pts_p10":    np.percentile(points, 10, axis=0),
        "pts_p90":    np.percentile(points, 90, axis=0),
        "gf_medios":  gf.mean(axis=0),
        "gc_medios":  ga.mean(axis=0),
        "pos_media":  positions.mean(axis=0),
        "p_titulo":   (positions == 1).mean(axis=0),
        "p_top4":     (positions <= CHAMPIONS_SPOTS).mean(axis=0),
        "p_top6":     (positions <= 6).mean(axis=0),
        "p_descenso": (positions > n_teams - RELEGATION_SPOTS).mean(axis=0),
    }).sort_values("pos_media").reset_index(drop=True)

    return SeasonResults(
        table=table, points=points, positions=positions,
        goals_for=gf, goals_against=ga,
        teams=teams, n_sims=n_sims,
        n_simulated_matches=n_fix, n_played_matches=n_played,
    )


def _rank(points, gd, gf, rng) -> np.ndarray:
    """
    Posición final por simulación con los desempates de la Premier:
    puntos, diferencia de goles, goles a favor, y aleatorio como último recurso.

    np.lexsort ordena por la ÚLTIMA clave primero, de ahí el orden invertido.
    """
    n_sims, n_teams = points.shape
    noise = rng.random((n_sims, n_teams))
    order = np.lexsort((noise, -gf, -gd, -points), axis=1)   # mejores primero
    positions = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    positions[rows, order] = np.arange(1, n_teams + 1)[None, :]
    return positions


# ── 3. Exportación ───────────────────────────────────────────────────────────
def export_results(
    res: SeasonResults,
    fixtures_pred: pd.DataFrame | None = None,
    outdir: Path | None = None,
    save_raw: bool = False,
) -> dict[str, Path]:
    """
    Vuelca los resultados a CSV.

    Ficheros generados en results/:

      season_probabilities.csv  tabla agregada, una fila por equipo
      match_predictions.csv     un partido por fila con sus probabilidades
      position_matrix.csv       P(equipo termine en puesto N), 20x20
      points_distribution.csv   histograma de puntos finales por equipo
      raw_points.csv            (opcional) puntos crudos de cada simulación

    `raw_points.csv` sólo con save_raw=True: con 20.000 simulaciones son 20.000
    filas × 20 columnas, útil para análisis propios pero pesado.
    """
    outdir = Path(outdir or RESULTS_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    written = {}

    p = outdir / "season_probabilities.csv"
    res.table.to_csv(p, index=False); written["season_probabilities"] = p

    p = outdir / "position_matrix.csv"
    res.position_matrix().to_csv(p); written["position_matrix"] = p

    p = outdir / "points_distribution.csv"
    res.points_distribution().to_csv(p, index=False); written["points_distribution"] = p

    if fixtures_pred is not None:
        p = outdir / "match_predictions.csv"
        fixtures_pred.to_csv(p, index=False); written["match_predictions"] = p

    if save_raw:
        p = outdir / "raw_points.csv"
        pd.DataFrame(res.points, columns=res.teams).to_csv(p, index=False)
        written["raw_points"] = p

    for nombre, ruta in written.items():
        logger.info(f"  {nombre:<22} -> {ruta}")

    return written
