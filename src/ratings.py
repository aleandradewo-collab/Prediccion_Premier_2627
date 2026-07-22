"""
ratings.py - Motor de ratings de ataque y defensa (Dixon-Coles).

Modelo multiplicativo de goles esperados:

    lambda_local     = mu * ataque[local] * defensa[visitante] * ventaja_local
    lambda_visitante = mu * ataque[visitante] * defensa[local]

donde ataque > 1 significa "marca más que la media" y defensa < 1 significa
"encaja menos que la media".

Tres diferencias respecto a la versión del Mundial:

1. VECTORIZADO. El ajuste alternado usa `np.bincount` en lugar de bucles con
   `iterrows`. Necesario porque el backtest recalcula ratings jornada a jornada
   sobre 21 temporadas: con bucles serían horas, así son segundos.

2. CORTE TEMPORAL. `fit_ratings(..., as_of=fecha)` usa exclusivamente partidos
   anteriores a esa fecha. Sin esto el backtest se contamina con información
   del futuro y arroja métricas artificialmente buenas.

3. CORRECCIÓN TAU. El Poisson bivariante independiente infraestima los
   marcadores bajos (0-0, 1-0, 0-1, 1-1). En una liga inglesa esos resultados
   son muy frecuentes, así que la corrección de Dixon-Coles (1997) cambia de
   forma apreciable las probabilidades de empate y, acumulada sobre 38
   jornadas, la tabla final.

La vida media del decaimiento temporal es MUCHO más corta que en el Mundial
(meses en lugar de años): en liga la fuerza de un equipo cambia dentro de la
propia temporada por fichajes, lesiones y cambios de entrenador.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

from src.utils import logger

# ── Parámetros por defecto ────────────────────────────────────────────────────
HALF_LIFE_DAYS  = 240     # vida media del peso temporal (~8 meses)
PRIOR_STRENGTH  = 4.0     # "goles fantasma" que empujan los ratings hacia 1.0
MAX_ITER        = 200
TOL             = 1e-9
MAX_GOALS       = 10      # truncado de la matriz de marcadores
LAMBDA_MIN      = 0.15
LAMBDA_MAX      = 6.0

# ── Valores por defecto para equipos sin datos ────────────────────────────────
# Medidos, no inventados: mediana de los 60 equipos ascendidos entre 2006/07 y
# 2025/26, calculada sobre su PRIMERA temporada en Primera.
#
#              ataque   defensa    net
#   mediana     0.768     1.196    0.642
#   media       0.787     1.196    0.658
#
# Recordatorio: la media de la liga es 1.000 en ambos ratings, y en defensa
# MENOS es mejor. Un ascendido típico marca un 23% menos y encaja un 20% más
# que el equipo medio.
#
# Sin esto, un equipo sin histórico recibe rating neutro (1.0) y el modelo lo
# coloca a mitad de tabla. Es el caso de Hull, cuyo peso efectivo es 0.002
# porque su última temporada en Primera fue 2016/17 y el decaimiento temporal
# la ha borrado, pero que aun así salía 14º.
DEFAULT_ATTACK  = 0.768
DEFAULT_DEFENSE = 1.196

# Peso efectivo (suma de pesos temporales de sus partidos) a partir del cual se
# considera que un equipo tiene historia suficiente para fiarse de su rating.
# Por debajo, se mezcla proporcionalmente con el default.
# Referencia: un equipo con temporada completa reciente ronda 31.
FULL_WEIGHT     = 25.0


@dataclass
class Ratings:
    """Ratings ajustados a una fecha concreta."""
    attack:    dict[str, float]
    defense:   dict[str, float]
    home_adv:  float
    mu:        float
    rho:       float
    as_of:     pd.Timestamp
    n_matches: int
    eff_weight: dict[str, float] = field(default_factory=dict)

    def teams(self) -> list[str]:
        return sorted(self.attack)

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "team":        self.teams(),
            "attack":      [self.attack[t]  for t in self.teams()],
            "defense":     [self.defense[t] for t in self.teams()],
            "eff_weight":  [self.eff_weight.get(t, np.nan) for t in self.teams()],
        })
        df["net"] = df["attack"] / df["defense"]
        return df.sort_values("net", ascending=False).reset_index(drop=True)


# ── Corrección de Dixon-Coles ─────────────────────────────────────────────────
def dc_tau(x, y, lh, la, rho):
    """
    Factor de corrección para marcadores bajos.

    Sólo afecta a las cuatro celdas donde el Poisson independiente falla:
    0-0, 0-1, 1-0 y 1-1. El resto de la matriz queda intacta.
    """
    x = np.asarray(x); y = np.asarray(y)
    lh = np.asarray(lh, dtype=float); la = np.asarray(la, dtype=float)

    tau = np.ones(np.broadcast(x, y, lh, la).shape, dtype=float)
    tau = np.where((x == 0) & (y == 0), 1.0 - lh * la * rho, tau)
    tau = np.where((x == 0) & (y == 1), 1.0 + lh * rho,      tau)
    tau = np.where((x == 1) & (y == 0), 1.0 + la * rho,      tau)
    tau = np.where((x == 1) & (y == 1), 1.0 - rho,           tau)
    return tau


def _time_weights(dates: pd.Series, as_of: pd.Timestamp, half_life_days: float) -> np.ndarray:
    """Peso exponencial: los partidos recientes valen más."""
    days = (as_of - dates).dt.total_seconds().to_numpy() / 86400.0
    days = np.maximum(days, 0.0)
    return np.exp(-np.log(2.0) * days / half_life_days)


# ── Ajuste ────────────────────────────────────────────────────────────────────
def fit_ratings(
    matches: pd.DataFrame,
    as_of: pd.Timestamp | str | None = None,
    half_life_days: float = HALF_LIFE_DAYS,
    prior_strength: float = PRIOR_STRENGTH,
    fit_rho: bool = False,
    max_iter: int = MAX_ITER,
    tol: float = TOL,
    verbose: bool = False,
) -> Ratings:
    """
    Ajusta ataque, defensa, ventaja de local y rho usando sólo los partidos
    anteriores a `as_of`.

    Args:
        matches:        DataFrame con date, home, away, home_goals, away_goals
        as_of:          fecha de corte. None = usar todo el histórico
        half_life_days: vida media del decaimiento temporal
        prior_strength: encogimiento hacia la media de la liga. Sube este valor
                        si tienes pocos partidos por equipo (inicio de temporada,
                        recién ascendidos)
        fit_rho:        corrección de Dixon-Coles. Por defecto False:
                        el backtest muestra que no aporta (ver README)
    """
    if as_of is None:
        as_of = matches["date"].max() + pd.Timedelta(days=1)
    as_of = pd.Timestamp(as_of)

    df = matches.loc[matches["date"] < as_of].copy()
    if df.empty:
        raise ValueError(f"No hay partidos anteriores a {as_of.date()}")

    w = _time_weights(df["date"], as_of, half_life_days)

    teams = sorted(set(df["home"]) | set(df["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    h = df["home"].map(idx).to_numpy()
    a = df["away"].map(idx).to_numpy()
    hg = df["home_goals"].to_numpy(dtype=float)
    ag = df["away_goals"].to_numpy(dtype=float)

    # Media global ponderada de goles por equipo y partido
    mu = float(np.average(np.concatenate([hg, ag]), weights=np.concatenate([w, w])))

    attack   = np.ones(n)
    defense  = np.ones(n)
    home_adv = 1.30

    for it in range(max_iter):
        atk_old, def_old, ha_old = attack.copy(), defense.copy(), home_adv

        # ── Ataque ────────────────────────────────────────────────────────────
        # Numerador: goles marcados ponderados. Denominador: goles que se
        # esperarían con ataque neutro. El cociente es el rating.
        num = np.bincount(h, weights=w * hg, minlength=n) + \
              np.bincount(a, weights=w * ag, minlength=n)
        den = np.bincount(h, weights=w * mu * defense[a] * home_adv, minlength=n) + \
              np.bincount(a, weights=w * mu * defense[h],            minlength=n)
        attack = (num + prior_strength) / (den + prior_strength)

        # ── Defensa ───────────────────────────────────────────────────────────
        num = np.bincount(h, weights=w * ag, minlength=n) + \
              np.bincount(a, weights=w * hg, minlength=n)
        den = np.bincount(h, weights=w * mu * attack[a],            minlength=n) + \
              np.bincount(a, weights=w * mu * attack[h] * home_adv, minlength=n)
        defense = (num + prior_strength) / (den + prior_strength)

        # ── Normalización a media 1 ───────────────────────────────────────────
        attack  /= attack.mean()
        defense /= defense.mean()

        # ── Ventaja de local ──────────────────────────────────────────────────
        exp_home = np.sum(w * mu * attack[h] * defense[a])
        home_adv = float(np.sum(w * hg) / exp_home) if exp_home > 0 else ha_old
        home_adv = float(np.clip(home_adv, 1.0, 2.0))

        delta = max(np.abs(attack - atk_old).max(),
                    np.abs(defense - def_old).max(),
                    abs(home_adv - ha_old))
        if delta < tol:
            break

    # Recalibrar mu para que el total de goles esperado case con el observado
    exp_total = np.sum(w * mu * attack[h] * defense[a] * home_adv) + \
                np.sum(w * mu * attack[a] * defense[h])
    obs_total = np.sum(w * (hg + ag))
    if exp_total > 0:
        mu *= obs_total / exp_total

    lh = np.clip(mu * attack[h] * defense[a] * home_adv, LAMBDA_MIN, LAMBDA_MAX)
    la = np.clip(mu * attack[a] * defense[h],            LAMBDA_MIN, LAMBDA_MAX)

    rho = _fit_rho(hg, ag, lh, la, w) if fit_rho else 0.0

    eff_w = np.bincount(h, weights=w, minlength=n) + np.bincount(a, weights=w, minlength=n)

    if verbose:
        logger.info(
            f"  ratings @ {as_of.date()}: {len(df):,} partidos, {n} equipos, "
            f"{it + 1} iter, mu={mu:.3f}, local={home_adv:.3f}, rho={rho:+.4f}"
        )

    return Ratings(
        attack=dict(zip(teams, attack)),
        defense=dict(zip(teams, defense)),
        home_adv=home_adv,
        mu=mu,
        rho=rho,
        as_of=as_of,
        n_matches=len(df),
        eff_weight=dict(zip(teams, eff_w)),
    )


def _fit_rho(hg, ag, lh, la, w) -> float:
    """
    Estima rho por máxima verosimilitud, manteniendo ataque y defensa fijos.

    Dixon-Coles lo ajustan conjuntamente; separarlo es una aproximación, pero
    el efecto de rho sobre ataque y defensa es pequeño y así el ajuste queda
    mucho más simple y rápido.
    """
    def neg_ll(rho):
        tau = dc_tau(hg, ag, lh, la, rho)
        if np.any(tau <= 0):
            return 1e10
        return -float(np.sum(w * np.log(tau)))

    # Cota que garantiza tau > 0 en las cuatro celdas corregidas
    bound = min(0.99, 1.0 / max(float(np.max(lh * la)), 1e-9)) * 0.95
    res = minimize_scalar(neg_ll, bounds=(-bound, bound), method="bounded")
    return float(res.x) if res.success else 0.0


# ── Defaults para equipos sin datos ───────────────────────────────────────────
def apply_defaults(
    ratings: Ratings,
    teams: list[str] | None = None,
    default_attack: float = DEFAULT_ATTACK,
    default_defense: float = DEFAULT_DEFENSE,
    full_weight: float = FULL_WEIGHT,
    verbose: bool = False,
) -> Ratings:
    """
    Rellena y corrige ratings de equipos con poca o ninguna historia.

    Dos casos:

    1. Equipo AUSENTE (nunca jugó en el periodo del histórico, p.ej. Coventry):
       recibe directamente los valores por defecto.

    2. Equipo con HISTORIA INSUFICIENTE (p.ej. Hull, peso efectivo 0.002):
       su rating ajustado se mezcla con el default en proporción a cuánta
       información real tiene.

           w = min(1, peso_efectivo / full_weight)
           rating = w * ajustado + (1 - w) * default

       Un equipo con temporada completa reciente tiene w = 1 y no se toca.
       Uno sin datos tiene w = 0 y queda en el default. En medio, transición
       suave en lugar de un salto brusco.

    Args:
        teams: equipos que deben existir sí o sí en la salida. Normalmente los
               20 de la temporada a simular.
    """
    attack  = dict(ratings.attack)
    defense = dict(ratings.defense)
    eff     = dict(ratings.eff_weight)

    objetivo = set(teams) if teams is not None else set(attack)
    ajustes = []

    for t in sorted(objetivo):
        ew = eff.get(t, 0.0)
        w = min(1.0, ew / full_weight) if full_weight > 0 else 1.0

        if t not in attack:
            attack[t], defense[t], eff[t] = default_attack, default_defense, 0.0
            ajustes.append((t, ew, 0.0, default_attack, default_defense, "ausente"))
            continue

        if w >= 1.0:
            continue

        a0, d0 = attack[t], defense[t]
        attack[t]  = w * a0 + (1 - w) * default_attack
        defense[t] = w * d0 + (1 - w) * default_defense
        ajustes.append((t, ew, w, attack[t], defense[t], "mezclado"))

    if verbose and ajustes:
        logger.info(f"  defaults aplicados a {len(ajustes)} equipos:")
        for t, ew, w, a, d, tipo in ajustes:
            logger.info(f"    {t:16s} peso={ew:6.2f} w={w:.2f} -> "
                        f"atk={a:.3f} def={d:.3f}  [{tipo}]")

    return Ratings(
        attack=attack, defense=defense,
        home_adv=ratings.home_adv, mu=ratings.mu, rho=ratings.rho,
        as_of=ratings.as_of, n_matches=ratings.n_matches, eff_weight=eff,
    )


# ── Predicción ────────────────────────────────────────────────────────────────
def predict_lambdas(ratings: Ratings, home: str, away: str,
                    neutral: bool = False) -> tuple[float, float]:
    """
    Goles esperados de un enfrentamiento.

    Un equipo no presente en los ratings recibe los valores por defecto de
    ascendido, NO valores neutros: si no sabemos nada de un equipo, lo más
    probable es que venga de Segunda, no que sea de media tabla.
    """
    atk_h = ratings.attack.get(home, DEFAULT_ATTACK)
    def_h = ratings.defense.get(home, DEFAULT_DEFENSE)
    atk_a = ratings.attack.get(away, DEFAULT_ATTACK)
    def_a = ratings.defense.get(away, DEFAULT_DEFENSE)
    ha = 1.0 if neutral else ratings.home_adv

    lh = float(np.clip(ratings.mu * atk_h * def_a * ha, LAMBDA_MIN, LAMBDA_MAX))
    la = float(np.clip(ratings.mu * atk_a * def_h,      LAMBDA_MIN, LAMBDA_MAX))
    return lh, la


def score_matrix(lh: float, la: float, rho: float = 0.0,
                 max_goals: int = MAX_GOALS) -> np.ndarray:
    """
    Matriz de probabilidad de cada marcador, con la corrección tau aplicada
    y renormalizada para que sume 1.
    """
    g = np.arange(max_goals + 1)
    ph = poisson.pmf(g, lh)
    pa = poisson.pmf(g, la)
    m = np.outer(ph, pa)

    if rho != 0.0:
        x, y = np.meshgrid(g, g, indexing="ij")
        m = m * dc_tau(x, y, lh, la, rho)
        m = np.clip(m, 0.0, None)

    return m / m.sum()


def match_probabilities(lh: float, la: float, rho: float = 0.0,
                        max_goals: int = MAX_GOALS) -> dict:
    """Probabilidades 1X2 y puntos esperados a partir de la matriz de marcadores."""
    m = score_matrix(lh, la, rho, max_goals)
    p_home = float(np.tril(m, -1).sum())   # filas > columnas
    p_draw = float(np.trace(m))
    p_away = float(np.triu(m, 1).sum())
    return {
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "xpts_home": 3 * p_home + p_draw,
        "xpts_away": 3 * p_away + p_draw,
        "lambda_home": lh, "lambda_away": la,
    }


def predict_match(ratings: Ratings, home: str, away: str, **kw) -> dict:
    """Atajo: ratings + equipos -> probabilidades."""
    lh, la = predict_lambdas(ratings, home, away, neutral=kw.pop("neutral", False))
    return {"home": home, "away": away,
            **match_probabilities(lh, la, ratings.rho, **kw)}
