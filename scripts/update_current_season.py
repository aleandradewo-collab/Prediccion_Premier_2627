"""
update_current_season.py — Refresca SOLO la temporada en curso, directo de
football-data.co.uk (no del espejo de datasets/football-datasets que usa
download_footballdata.py, que puede tardar en sincronizar durante la
temporada).

Pensado para correr en un GitHub Action programado: el entorno de
desarrollo de este proyecto tiene bloqueado el acceso a football-data.co.uk
por política de red, pero un runner de GitHub Actions no. Ver
.github/workflows/update_results.yml.

NO VERIFICADO EN VIVO — sin acceso a football-data.co.uk no hay forma de
probar esto desde aquí. El patrón de URL (mmz4281/{código}/E0.csv) y el
formato de columnas son los mismos que usa el resto del ecosistema de
football-data.co.uk desde hace años, pero conviene disparar el workflow
manualmente (workflow_dispatch) una vez y revisar el resultado antes de
confiar en el cron.

Reemplaza en epl_matches.csv las filas de la temporada `--season` por la
versión recién descargada; el resto del fichero queda intacto. Aborta sin
tocar nada si las fechas no parsean o si trae MENOS partidos de los que ya
había guardados — mejor fallar alto que perder datos en silencio.

Uso:
    python scripts/update_current_season.py                  # temporada 2026 (2026/27)
    python scripts/update_current_season.py --season 2025
    python scripts/update_current_season.py --out data/raw/epl_matches.csv
"""

import argparse
import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))
from download_footballdata import CORE_COLS, KEEP_COLS, season_code, season_label

DIRECT_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"


def fetch_current(start_year: int, timeout: int = 30) -> pd.DataFrame:
    """
    Descarga la temporada `start_year` directo de football-data.co.uk.

    A diferencia del espejo (ISO), football-data.co.uk publica las fechas en
    formato día/mes/año — se asume explícitamente (dayfirst=True), nunca se
    deja a pandas inferir el formato.
    """
    url = DIRECT_URL.format(code=season_code(start_year))
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    if df.empty:
        sys.exit(f"ERROR: {url} no trajo ningún partido con equipos válidos.")

    df.insert(0, "season", season_label(start_year))
    df.insert(1, "season_start", start_year)

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    bad = df["Date"].isna().sum()
    if bad:
        sys.exit(f"ERROR: {bad} fechas de {url} no parsearon con dayfirst=True. "
                 f"¿Cambió el formato de fecha de football-data.co.uk? Revisar antes de continuar.")

    return df[[c for c in KEEP_COLS if c in df.columns]]


def merge_into(existing_path: Path, new: pd.DataFrame, season: str) -> pd.DataFrame:
    existing = pd.read_csv(existing_path)
    existing["Date"] = pd.to_datetime(existing["Date"], format="%Y-%m-%d")

    old_count = int((existing["season"] == season).sum())
    if len(new) < old_count:
        sys.exit(f"ERROR: la descarga trae {len(new)} partidos de {season}, "
                 f"{existing_path} ya tenía {old_count}. Abortando sin tocar el fichero — "
                 f"esto casi seguro es un fallo de la fuente, no una temporada que haya "
                 f"perdido partidos.")

    merged = pd.concat([existing[existing["season"] != season], new], ignore_index=True)
    return merged.sort_values("Date").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", type=int, default=2026,
                   help="Año de inicio de la temporada a refrescar (default: 2026, ie. 2026/27)")
    p.add_argument("--out", type=Path, default=Path("data/raw/epl_matches.csv"))
    args = p.parse_args()

    label = season_label(args.season)
    print(f"Descargando {label} directo de football-data.co.uk...")
    new = fetch_current(args.season)

    nulls = new[CORE_COLS].isna().sum().sum()
    if nulls:
        sys.exit(f"ERROR: {nulls} nulos en columnas obligatorias tras la descarga.")

    if not args.out.exists():
        sys.exit(f"ERROR: {args.out} no existe — este script sólo refresca una temporada "
                 f"dentro de un histórico ya construido. Usa download_footballdata.py primero.")

    merged = merge_into(args.out, new, label)
    merged.to_csv(args.out, index=False)

    print(f"  {label}: {len(new)} partidos ({new['Date'].min().date()} -> {new['Date'].max().date()})")
    print(f"  Guardado en {args.out} ({len(merged):,} partidos en total)")


if __name__ == "__main__":
    main()
