# Premier League 2026/27 Predictor

Predicción de la temporada 2026/27 de la Premier League mediante un modelo de goles
Dixon-Coles con ratings dinámicos y simulación Monte Carlo de la temporada completa.

La salida no es "gana el Arsenal", sino una distribución de probabilidad: título,
top-4, descenso y puntos esperados para cada equipo, más las probabilidades 1X2 de
los 380 partidos del calendario.

Sucesor del [World Cup 2026 Predictor](https://github.com/aleandradewo-collab/Prediccion_mundial_2026),
adaptado de formato copa a formato liga.

---

## Qué cambia respecto al modelo del Mundial

| | Mundial 2026 | Premier League 2026/27 |
|---|---|---|
| Partidos | 64, con eliminatorias | 380, todos contra todos |
| Fuerza de equipo | Prácticamente constante | Cambia por fichajes, lesiones y cambios de entrenador |
| Calendario | Sin congestión relevante | Champions, Europa League, FA Cup, Carabao |
| Desempate | Binario (pasas o no) | Puntos, diferencia de goles, goles a favor |
| Equipos nuevos | Ninguno | 3 ascendidos sin histórico en la categoría |
| Vida media del decaimiento | 3 años | 240 días |

---

## Estado del proyecto

- [x] Ingesta de resultados históricos (2005/06 – 2025/26)
- [x] Unificación de nomenclaturas entre fuentes
- [x] Definición de los 20 equipos de 2026/27
- [x] Motor de ratings ataque/defensa vectorizado, sin fuga temporal
- [x] Defaults para equipos sin datos
- [x] Simulador Monte Carlo de temporada completa
- [x] Predicción partido a partido y por jornada
- [x] Simulación condicionada a resultados ya jugados
- [ ] Features de plantilla y fichajes
- [ ] Feature de congestión de calendario (Europa y copas)
- [ ] Predicciones individuales (Bota de Oro, asistencias)

---

## Instalación

```bash
git clone https://github.com/<usuario>/Prediccion_Premier_2627.git
cd Prediccion_Premier_2627
python -m pip install -r requirements.txt
```

---

## Ejecutar el modelo

### Paso 1 — Ingesta de datos

```bash
python scripts/download_footballdata.py
```

Descarga las temporadas de la Premier y las consolida en `data/raw/epl_matches.csv`.
Fuente: [datasets/football-datasets](https://github.com/datasets/football-datasets),
espejo de [football-data.co.uk](https://www.football-data.co.uk/) actualizado a diario
mediante GitHub Actions. Aborta si alguna fecha no parsea y avisa si una temporada no
tiene 380 partidos.

| Opción | Efecto |
|---|---|
| `--from-season 2015` | Empezar en esa temporada en vez de 2005 |
| `--to-season 2027` | Última temporada a intentar (omite las inexistentes) |
| `--out ruta.csv` | Cambiar el destino |

**Durante la temporada 2026/27 basta con reejecutarlo** para incorporar las jornadas
nuevas. El resto de ficheros de `data/raw/` son estáticos y ya están en el repositorio.

### Paso 2 — Validar el motor de ratings

```bash
python scripts/backtest_ratings.py            # métricas del modelo
python scripts/backtest_ratings.py --sweep    # calibrar la vida media
```

Recorre las temporadas jornada a jornada reajustando los ratings con información
exclusivamente anterior a cada partido, y calcula log-loss, RPS, Brier y calibración
de empates. Es el fichero que dice si un cambio mejora de verdad o solo lo parece.

| Opción | Efecto |
|---|---|
| `--start-season 2015-16` | Primera temporada evaluada |
| `--half-life 240` | Vida media del decaimiento a probar |
| `--prior 4` | Encogimiento hacia la media de la liga |
| `--sweep` | Barrido de vidas medias con comparativa |

### Paso 3 — Predecir la temporada

```bash
python scripts/simulate_season.py
```

Simula 20.000 temporadas completas y genera todos los ficheros de resultados.
Tarda unos 2 segundos.

| Opción | Efecto |
|---|---|
| `--sims 50000` | Número de temporadas simuladas. **Menos de 5.000 es ruido** |
| `--matchday 1` | Sólo esa jornada, probabilidades analíticas |
| `--date 2026-08-22` | Sólo los partidos de esa fecha |
| `--played fichero.csv` | Incorporar resultados ya disputados |
| `--as-of 2026-12-01` | Ratings calculados a otra fecha |
| `--no-defaults` | Sin el prior de equipos ascendidos |
| `--save-raw` | Guardar también los puntos de cada simulación |

#### Predicción de una jornada

Con `--matchday` o `--date` no se lanza Monte Carlo: se calculan las probabilidades
**analíticamente** a partir de la matriz de marcadores. Son exactas, sin ruido de
muestreo, e instantáneas.

```bash
python scripts/simulate_season.py --matchday 1
```

> La columna de marcador muestra casi siempre `1-1` o `2-0`. No es un error: con
> lambdas alrededor de 1,3-1,5 el resultado individual más probable es casi siempre
> 1-1, aunque su probabilidad ronde el 10%. Es la moda de una distribución muy plana.
> Para leer un partido, mira las columnas 1X2.

#### Simulación durante la temporada

Con `--played` se incorporan los partidos ya disputados: sus puntos y goles se suman
como base y sólo se simula lo que queda. Es el modo que usarás cada jornada.

```bash
python scripts/simulate_season.py --played results/jugados.csv
```

El CSV debe tener las columnas `home`, `away`, `home_goals`, `away_goals`, con los
nombres en nomenclatura canónica (la de `team_names.csv`).

Ejemplo del efecto — Arsenal perdiendo sus 5 primeros partidos:

| | Sin jugar | Tras 5 derrotas |
|---|---|---|
| Arsenal, título | 49,1% | 12,1% |
| Arsenal, top-4 | 96,9% | 71,3% |
| Man City, título | 42,8% | 67,5% |

---

## Ficheros de resultados

Todo se escribe en `results/`, que está en `.gitignore`.

| Fichero | Filas | Contenido |
|---|---|---|
| `season_probabilities.csv` | 20 | Puntos medios, P10/P90, goles, P(título), P(top-4), P(top-6), P(descenso) |
| `match_predictions.csv` | 380 | Un partido por fila: lambdas, 1X2, xPts, marcador más probable |
| `position_matrix.csv` | 20×20 | P(cada equipo termine en cada puesto) — ideal para un heatmap |
| `points_distribution.csv` | 400 | Histograma de puntos finales por equipo |
| `raw_points.csv` | n_sims | Sólo con `--save-raw`. Puntos de cada simulación, para análisis propios |

---

## Uso desde Python

```python
from src.utils     import load_matches, load_fixtures, load_teams_2026_27
from src.ratings   import fit_ratings, apply_defaults, predict_match
from src.simulator import simulate_season, predict_fixtures, export_results

matches  = load_matches()
fixtures = load_fixtures()
teams    = load_teams_2026_27()

# Ratings al inicio de temporada, con el prior de ascendidos aplicado
r = fit_ratings(matches, as_of="2026-08-21")
r = apply_defaults(r, teams=list(teams["canonical"]), verbose=True)

print(r.to_frame())                              # tabla de ratings

# Un partido concreto
print(predict_match(r, "Arsenal", "Coventry"))

# Todos los partidos del calendario, sin Monte Carlo
pred = predict_fixtures(r, fixtures)

# Temporada completa
res = simulate_season(r, fixtures, n_sims=20_000)
print(res.summary())
print(res.position_matrix())          # P(puesto) por equipo
print(res.points_distribution())      # histograma de puntos

export_results(res, fixtures_pred=pred)
```

Una jornada suelta:

```python
j1 = fixtures[fixtures["matchday"] == 1]
predict_fixtures(r, j1)
```

Condicionar a resultados ya jugados:

```python
import pandas as pd
jugados = pd.read_csv("results/jugados.csv")
res = simulate_season(r, fixtures, played=jugados, n_sims=20_000)
```

---

## Estructura

```
Prediccion_Premier_2627/
├── data/
│   ├── raw/
│   │   ├── epl_matches.csv                   # 7.980 partidos, 21 temporadas
│   │   ├── premier-league-gb-eng_2026-27.csv # calendario a predecir
│   │   ├── team_names.csv                    # mapeo entre las 3 nomenclaturas
│   │   ├── teams_2026_27.csv                 # los 20 equipos
│   │   ├── games.csv                         # Transfermarkt: PL + Europa + copas
│   │   ├── player_valuations.csv             # valores de mercado con histórico
│   │   ├── transfers.csv                     # movimientos de fichajes
│   │   ├── players.csv                       # metadatos de jugadores
│   │   └── competitions.csv                  # códigos de competición
│   └── processed/                            # generado — en .gitignore
├── src/                                      # librería
│   ├── __init__.py
│   ├── utils.py                              # rutas, carga y normalización
│   ├── ratings.py                            # motor Dixon-Coles
│   └── simulator.py                          # Monte Carlo y exportación
├── scripts/                                  # ejecutables
│   ├── download_footballdata.py              # ingesta reproducible
│   ├── backtest_ratings.py                   # validación del motor
│   └── simulate_season.py                    # predicción de la temporada
├── models/                                   # en .gitignore
├── results/                                  # en .gitignore
├── requirements.txt
└── README.md
```

**`src/` es librería, `scripts/` son ejecutables.** `simulator.py` define funciones que
otros módulos importan; `simulate_season.py` es lo que se lanza desde la terminal.

### Qué hace cada módulo

| Módulo | Rol |
|---|---|
| `utils.py` | **Consolidación.** Rutas y cargadores que normalizan las tres nomenclaturas, fuerzan formatos de fecha y validan integridad antes de que el modelo toque nada |
| `ratings.py` | **Modelado.** `fit_ratings()` ajusta ataque, defensa y ventaja de local; `apply_defaults()` corrige equipos sin historia; `predict_match()` da probabilidades 1X2 |
| `simulator.py` | **Simulación.** `predict_fixtures()` analítico por partido; `simulate_season()` Monte Carlo agregado; `export_results()` vuelca a CSV |

---

## Resultados

### Motor de ratings

Backtest sobre 4.180 partidos (2015/16 – 2025/26), jornada a jornada y sin fuga temporal:

| Métrica | Modelo | Predicción uniforme |
|---|---|---|
| log-loss | **0,9755** | 1,0986 |
| RPS | **0,2004** | 0,2222 |
| Acierto | 53,1% | 33,3% |
| Empates predichos | 23,6% | — |
| Empates reales | 23,7% | — |

Ajuste completo sobre 7.980 partidos: **0,01 s**. Backtest completo: 16 s.
Simulación de 20.000 temporadas: **1,1 s**.

### Calibración de la vida media

| Vida media | RPS | | Vida media | RPS |
|---|---|---|---|---|
| 90 d | 0,2032 | | 365 d | 0,2012 |
| 120 d | 0,2021 | | 540 d | 0,2020 |
| 180 d | 0,2012 | | 730 d | 0,2030 |
| **240 d** | **0,2010** | | 1.095 d | 0,2043 |

Curva en U limpia con óptimo en 240 días. Los 3 años del modelo del Mundial (1.095 días)
quedan claramente peor, lo que confirma que en liga la fuerza de un equipo cambia en
escala de meses.

### Proyección 2026/27 (sólo ratings históricos)

```
  #   Equipo              Pts    P10-P90   Título    Top-4    Desc.
  1   Arsenal            77.5      68-87    49.1%    96.9%     0.0%
  2   Man City           76.4      67-86    42.8%    95.3%     0.0%
  3   Liverpool          64.0      54-74     3.9%    53.7%     0.1%
  4   Man United         59.6      49-70     1.3%    31.1%     0.3%
  5   Aston Villa        56.9      47-67     0.5%    20.0%     0.9%
  ...
  18  Hull               33.7      25-43     0.0%     0.0%    73.7%
  19  Coventry           33.6      25-43     0.0%     0.0%    74.5%
  20  Ipswich            32.5      23-42     0.0%     0.0%    79.3%
```

> **Esta proyección aún no incorpora fichajes, plantillas ni congestión de calendario.**
> Refleja el rendimiento de la temporada pasada, no el mercado de este verano. Los
> bloques pendientes son precisamente los que la convertirán en una predicción real
> de 2026/27.

---

## Notas metodológicas

### Corrección tau de Dixon-Coles: descartada

El paper original de 1997 corrige la infraestimación de marcadores bajos (0-0, 1-0,
0-1, 1-1). En este dataset **no aporta nada**:

| | log-loss | RPS | Empates predichos |
|---|---|---|---|
| Sin tau | 0,9771 | 0,2009 | 23,6% |
| Con tau | 0,9778 | 0,2010 | 24,2% |

El modelo ya calibra los empates casi exactamente (23,6% predicho frente a 23,7% real)
y tau los sobrestima. El código sigue disponible con `fit_rho=True`, pero el valor por
defecto es `False`.

### Valores por defecto para equipos sin datos

Un equipo sin histórico recibiría rating neutro (1,0) y el modelo lo colocaría a mitad
de tabla. Los defaults se midieron sobre los **60 equipos ascendidos entre 2006/07 y
2025/26**, evaluando su primera temporada en Primera:

| | Ataque | Defensa | Net |
|---|---|---|---|
| Mediana | 0,768 | 1,196 | 0,642 |
| Media | 0,787 | 1,196 | 0,658 |

Se aplican según el `peso efectivo` (suma de pesos temporales de los partidos de un
equipo; una temporada completa reciente ronda 31):

- **Equipo ausente** del histórico → recibe el default directamente.
- **Equipo con poca historia** → mezcla proporcional:
  `w = min(1, peso / 25)`, `rating = w · ajustado + (1−w) · default`.

Efecto en el backtest: log-loss 0,9766 → 0,9755 y RPS 0,2008 → 0,2004. La mejora es
pequeña porque en el histórico casi todos los ascendidos habían pisado la Premier antes.
El caso que el backtest **no puede medir** es el de Coventry, ausente por completo, que
es justamente donde el default resulta imprescindible.

### Equipos ascendidos en 2026/27

| Equipo | Peso efectivo | Tratamiento |
|---|---|---|
| Coventry | 0,00 | Default puro — 25 años fuera de Primera, sin registro en Transfermarkt |
| Hull | 0,002 | Default puro — última temporada 2016/17, borrada por el decaimiento |
| Ipswich | 7,09 | Mezcla al 28% con su rating de 2024/25 |

Pendiente: diferenciar el prior por vía de ascenso. Históricamente el campeón del
Championship rinde mejor que el ganador del playoff, y `teams_2026_27.csv` ya guarda esa
distinción en la columna `status`.

### Nomenclatura de equipos

Tres fuentes, tres convenciones: `Man United` (football-data), `Manchester United`
(calendario) y `Manchester United Football Club` (Transfermarkt). De 44 equipos solo uno
coincidía literalmente entre las tres. `team_names.csv` centraliza la traducción.

`load_fixtures()` incluye un guardarraíl que verifica que el calendario resuelve
exactamente a los 20 equipos de `teams_2026_27.csv` y detalla qué sobra y qué falta. Sin
él, un mapeo desalineado produce nombres válidos pero equivocados que no lanzan ningún
error y se propagan silenciosamente hasta la simulación.

### Fechas

Los CSV de football-data vienen en ISO (`YYYY-MM-DD`), no en formato británico. Tanto el
script de ingesta como `load_matches()` fuerzan el formato explícito y abortan si algo no
parsea. Dejándolo inferir, pandas invierte día y mes en silencio y ensucia el decaimiento
temporal sin dar ningún error.

### Número de simulaciones

Una liga de 38 jornadas tiene muchísimo ruido. Con una sola simulación el resultado es
prácticamente aleatorio. El mínimo utilizable son **5.000 temporadas**; el valor por
defecto es 20.000 y sigue tardando poco más de un segundo.

---

## Temporada 2026/27

Del **22 de agosto de 2026** al **30 de mayo de 2027**: 33 jornadas de fin de semana y 5
entre semana (jornadas 13, 18, 20, 25 y 28). El inicio y la última jornada se retrasaron
una semana por el Mundial 2026, lo que comprime la temporada y **aumenta la congestión**
respecto a un año normal. La ventana de fichajes de verano va del 15 de junio al 31 de
agosto de 2026.

**Descendidos de 2025/26:** Wolves (20º), Burnley (19º), West Ham (18º)
**Ascendidos:** Coventry City (campeón), Ipswich Town (2º), Hull City (playoff)

Arsenal defiende el título.

---

## Fuentes de datos

| Dataset | Fuente | Contenido |
|---|---|---|
| `epl_matches.csv` | football-data.co.uk vía datahub | Resultados, tiros, córners, tarjetas |
| `games.csv`, `players.csv`, `transfers.csv`, `player_valuations.csv` | [Transfermarkt / Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores) | Partidos de club, plantillas, valores de mercado |
| `premier-league-gb-eng_2026-27.csv` | Calendario oficial | 380 fixtures de la temporada |

`appearances.csv` (140 MB) no está en el repositorio por tamaño. Descárgalo del mismo
dataset de Kaggle si vas a usar el módulo de predicciones individuales.

---

## Licencia

MIT
