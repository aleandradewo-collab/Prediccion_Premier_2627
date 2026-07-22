"""
download_footballdata.py — Ingesta de resultados de la Premier League.

Fuente: https://github.com/datasets/football-datasets (espejo de football-data.co.uk,
actualizado a diario mediante GitHub Actions).

Genera data/raw/epl_matches.csv con una fila por partido.

Uso:
    python scripts/download_footballdata.py                  # desde 2005/06
    python scripts/download_footballdata.py --from-season 2015
    python scripts/download_footballdata.py --out data/raw/epl_matches.csv
"""

import argparse
import io
import sys
from pathlib import Path

import pandas as pd
import requests

BASE_URL = (
    "https://raw.githubusercontent.com/datasets/football-datasets/"
    "main/datasets/premier-league/season-{code}.csv"
)

# Columnas que conservamos, en orden. El resto se descarta.
KEEP_COLS = [
    "season", "season_start", "Date", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",          # resultado final
    "HTHG", "HTAG", "HTR",          # descanso
    "Referee",
    "HS", "AS", "HST", "AST",       # tiros / tiros a puerta
    "HF", "AF", "HC", "AC",         # faltas / corners
    "HY", "AY", "HR", "AR",         # tarjetas
]

CORE_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]


def season_code(start_year: int) -> str:
    """2024 -> '2425'.  1998 -> '9899'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """2024 -> '2024-25'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def fetch_season(start_year: int, timeout: int = 30) -> pd.DataFrame | None:
    """Descarga una temporada. Devuelve None si no existe todavia."""
    url = BASE_URL.format(code=season_code(start_year))
    resp = requests.get(url, timeout=timeout)

    if resp.status_code == 404:
        return None
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])

    df.insert(0, "season", season_label(start_year))
    df.insert(1, "season_start", start_year)
    return df


def build(from_season: int, to_season: int) -> pd.DataFrame:
    frames = []
    for year in range(from_season, to_season + 1):
        df = fetch_season(year)
        if df is None:
            print(f"  {season_label(year)}: no disponible todavia — se omite")
            continue
        print(f"  {season_label(year)}: {len(df)} partidos")
        frames.append(df)

    if not frames:
        sys.exit("ERROR: no se descargo ninguna temporada.")

    out = pd.concat(frames, ignore_index=True)

    # Las fechas del espejo vienen en ISO. Formato explicito para no
    # depender de la inferencia de pandas, que puede invertir dia y mes.
    out["Date"] = pd.to_datetime(out["Date"], format="%Y-%m-%d", errors="coerce")

    bad = out["Date"].isna().sum()
    if bad:
        sys.exit(f"ERROR: {bad} fechas no se pudieron parsear.")

    out = out.sort_values("Date").reset_index(drop=True)
    return out[[c for c in KEEP_COLS if c in out.columns]]


def validate(df: pd.DataFrame) -> None:
    """Comprobaciones minimas antes de escribir a disco."""
    nulls = df[CORE_COLS].isna().sum().sum()
    if nulls:
        sys.exit(f"ERROR: {nulls} nulos en columnas obligatorias.")

    counts = df.groupby("season").size()
    incomplete = counts[counts != 380]
    if len(incomplete):
        print("\n  AVISO — temporadas sin 380 partidos (normal si esta en curso):")
        for season, n in incomplete.items():
            print(f"    {season}: {n}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from-season", type=int, default=2005,
                   help="Anio de inicio de la primera temporada (default: 2005)")
    p.add_argument("--to-season", type=int, default=2026,
                   help="Anio de inicio de la ultima temporada (default: 2026)")
    p.add_argument("--out", type=Path, default=Path("data/raw/epl_matches.csv"))
    args = p.parse_args()

    print(f"Descargando Premier League {args.from_season}-{args.to_season}...")
    df = build(args.from_season, args.to_season)
    validate(df)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\n  {len(df):,} partidos · {df['season'].nunique()} temporadas "
          f"· {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print(f"  Guardado en {args.out}")


if __name__ == "__main__":
    main()
