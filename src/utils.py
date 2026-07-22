"""
utils.py - Rutas, logging y carga de datos del proyecto.
"""

import logging
from pathlib import Path

import pandas as pd

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT_DIR       = Path(__file__).parent.parent
DATA_RAW       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR     = ROOT_DIR / "models"
RESULTS_DIR    = ROOT_DIR / "results"

for _d in (DATA_PROCESSED, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("premier")


# ── Constantes de la competición ──────────────────────────────────────────────
N_TEAMS         = 20
N_MATCHDAYS     = 38
MATCHES_PER_SEASON = 380

# Códigos de competición en games.csv (Transfermarkt)
COMP_PREMIER    = "GB1"
COMP_EUROPE     = {"CL", "EL", "UCOL", "ELQ", "ECLQ"}   # UEFA
COMP_DOMESTIC_CUP = {"FAC", "CGB"}                       # FA Cup, Carabao


# ── Carga de datos ────────────────────────────────────────────────────────────
def load_matches(path: Path | None = None) -> pd.DataFrame:
    """
    Resultados históricos de la Premier.

    Devuelve un DataFrame normalizado con columnas:
        date, season, home, away, home_goals, away_goals
    más las estadísticas originales (tiros, córners, tarjetas).
    """
    path = path or DATA_RAW / "epl_matches.csv"
    df = pd.read_csv(path)

    # Formato explícito: los CSV vienen en ISO. Si se deja inferir, pandas
    # puede invertir día y mes en silencio y arruinar el decaimiento temporal.
    df["date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

    df = df.rename(columns={
        "HomeTeam": "home",
        "AwayTeam": "away",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
    })

    required = ["date", "home", "away", "home_goals", "away_goals"]
    missing = df[required].isna().sum().sum()
    if missing:
        raise ValueError(f"{missing} valores nulos en columnas obligatorias de {path}")

    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_fixtures(path: Path | None = None) -> pd.DataFrame:
    """
    Calendario 2026/27, con los nombres traducidos a la nomenclatura canónica.

    Devuelve: date, matchday, home, away
    """
    path = path or DATA_RAW / "premier-league-gb-eng_2026-27.csv"
    fx = pd.read_csv(path)
    fx["date"] = pd.to_datetime(fx["date"], format="%Y-%m-%d")

    names = load_team_names()
    # dropna sobre el DataFrame completo, no sobre una sola columna: hacer
    # zip(col.dropna(), otra_col) desalinea silenciosamente los pares, porque
    # la segunda columna conserva todas sus filas y zip trunca a la más corta.
    pares = names.dropna(subset=["fixture_name"])
    to_canon = dict(zip(pares["fixture_name"], pares["canonical"]))

    fx["home"] = fx["home_team"].map(to_canon)
    fx["away"] = fx["away_team"].map(to_canon)

    unmapped = set(fx.loc[fx["home"].isna(), "home_team"]) | \
               set(fx.loc[fx["away"].isna(), "away_team"])
    if unmapped:
        raise ValueError(
            f"Equipos del calendario sin traducir en team_names.csv: {sorted(unmapped)}"
        )

    # Guardarraíl: el calendario debe contener exactamente los 20 equipos de la
    # temporada. Un mapeo mal alineado produce nombres válidos pero equivocados,
    # que sin esta comprobación pasan desapercibidos hasta la simulación.
    en_calendario = set(fx["home"]) | set(fx["away"])
    if len(en_calendario) != N_TEAMS:
        raise ValueError(
            f"El calendario resuelve a {len(en_calendario)} equipos, se esperaban {N_TEAMS}"
        )
    try:
        esperados = set(load_teams_2026_27()["canonical"])
    except (FileNotFoundError, ValueError):
        esperados = None
    if esperados and en_calendario != esperados:
        raise ValueError(
            "Los equipos del calendario no coinciden con teams_2026_27.csv.\n"
            f"  sobran:  {sorted(en_calendario - esperados)}\n"
            f"  faltan:  {sorted(esperados - en_calendario)}"
        )

    return fx[["date", "matchday", "home", "away"]].sort_values("date").reset_index(drop=True)


def load_team_names(path: Path | None = None) -> pd.DataFrame:
    """Tabla de equivalencias entre football-data, el calendario y Transfermarkt."""
    return pd.read_csv(path or DATA_RAW / "team_names.csv")


def load_teams_2026_27(path: Path | None = None) -> pd.DataFrame:
    """Los 20 equipos de la temporada, con su estado (continua / ascendido)."""
    df = pd.read_csv(path or DATA_RAW / "teams_2026_27.csv")
    if len(df) != N_TEAMS:
        raise ValueError(f"Se esperaban {N_TEAMS} equipos, hay {len(df)}")
    return df


def league_table(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Clasificación a partir de un DataFrame de partidos jugados.

    Orden de desempate de la Premier: puntos, diferencia de goles, goles a favor.
    """
    rows = []
    for _, m in matches.iterrows():
        hg, ag = m["home_goals"], m["away_goals"]
        rows.append({"team": m["home"], "gf": hg, "ga": ag,
                     "pts": 3 if hg > ag else (1 if hg == ag else 0)})
        rows.append({"team": m["away"], "gf": ag, "ga": hg,
                     "pts": 3 if ag > hg else (1 if hg == ag else 0)})

    t = (pd.DataFrame(rows)
         .groupby("team")
         .agg(PJ=("pts", "size"), Pts=("pts", "sum"),
              GF=("gf", "sum"), GC=("ga", "sum"))
         .reset_index())
    t["DG"] = t["GF"] - t["GC"]
    t = t.sort_values(["Pts", "DG", "GF"], ascending=False).reset_index(drop=True)
    t.insert(0, "Pos", range(1, len(t) + 1))
    return t
