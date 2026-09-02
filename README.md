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
- [x] Features de plantilla y fichajes
- [x] Feature de congestión de calendario — medida y descartada, ver más abajo
- [x] Predicciones individuales (Bota de Oro, asistencias)

---

## Instalación

```bash
git clone https://github.com/<usuario>/Prediccion_Premier_2627.git
cd Prediccion_Premier_2627
python -m pip install -r requirements.txt
```

---

## Pipeline completo (referencia rápida)

Cada paso está explicado en detalle más abajo — esto es sólo la secuencia para
copiar y pegar. Los pasos 5-7 son independientes entre sí, en cualquier orden.

```bash
# 1. Dependencias
python -m pip install -r requirements.txt

# 2. appearances.csv en data/raw/ (no viene en el repo, ver Fuentes de datos)
#    Sólo hace falta para el paso 6 (predicciones individuales).

# 3. Refrescar resultados de la temporada en curso
python scripts/update_current_season.py
#    (o, si ya está fusionado a main, disparar el workflow de GitHub Actions
#    y hacer `git pull` después — ver Paso 2)

# 4. (Opcional) Validar que el motor sigue sano
python scripts/backtest_ratings.py

# 5. Predicción completa de la temporada
python scripts/simulate_season.py
#    o anclada a una jornada concreta, con lo ya jugado hasta ahí:
python scripts/simulate_season.py --through 5

# 6. Predicciones individuales (Bota de Oro, asistencias)
python scripts/predict_players.py

# 7. Cómo evolucionan las probabilidades jornada a jornada
python scripts/season_trajectory.py
```

Resultados en `results/` (CSV y Excel, `.gitignore`).

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

Es una ingesta completa desde cero: pensado para la primera vez, o para refrescar
todo el histórico. **Durante la temporada 2026/27, para incorporar sólo las jornadas
nuevas usa el Paso 2** — es más rápido y va directo a la fuente sin pasar por el espejo.
El resto de ficheros de `data/raw/` son estáticos y ya están en el repositorio.

### Paso 2 — Mantener los resultados al día

```bash
python scripts/update_current_season.py
```

Refresca **sólo** la temporada en curso en `epl_matches.csv`, directo de football-data.co.uk
(no del espejo de `datasets/football-datasets` que usa el Paso 1, que puede tardar en
sincronizar mientras la temporada está en marcha). Aborta sin tocar nada si las fechas no
parsean o si la descarga trae menos partidos de los que ya había guardados — nunca
sobrescribe con menos datos en silencio.

**Verificado en producción**: hay un workflow de GitHub Actions
(`.github/workflows/update_results.yml`) que lo ejecuta a diario y hace commit si hay
cambios — corre en la infraestructura de GitHub, que tiene salida a internet (a
diferencia del entorno donde se desarrolló este script, que la tiene bloqueada por
política de red, así que no pudo probarse ahí). Ya corrió solo y trajo resultados reales
de la temporada 2026/27 sin intervención. Si nunca lo disparaste a mano, hazlo una vez
(pestaña *Actions* → *Actualizar resultados de la temporada en curso* → *Run workflow*)
para confirmar que sigue funcionando en tu repo. Necesita permiso de escritura para
`GITHUB_TOKEN` (*Settings → Actions → General → Workflow permissions → Read and write*).

### Paso 3 — Validar el motor de ratings

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

### Paso 4 — Predecir la temporada

```bash
python scripts/simulate_season.py
```

Simula 20.000 temporadas completas y genera todos los ficheros de resultados —
CSV por separado y un `season_predictions.xlsx` con las mismas tablas, una por hoja.
Tarda unos 2 segundos.

| Opción | Efecto |
|---|---|
| `--sims 50000` | Número de temporadas simuladas. **Menos de 5.000 es ruido** |
| `--matchday 1` | Sólo esa jornada, probabilidades analíticas |
| `--date 2026-08-22` | Sólo los partidos de esa fecha |
| `--played fichero.csv` | Incorporar resultados ya disputados |
| `--through 5` | Predicción completa tras la jornada 5, detectando resultados solo (ver abajo) |
| `--as-of 2026-12-01` | Ratings calculados a otra fecha |
| `--no-defaults` | Sin el prior de equipos ascendidos |
| `--no-squad-prior` | Sin el prior de valor de plantilla (fichajes) |
| `--no-congestion` | No cargar/aplicar el ajuste de congestión de calendario (inerte por defecto, ver nota metodológica) |
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

#### Simulación durante la temporada, sin CSV manual

`--through N` hace lo mismo que `--played`, pero detectando solo qué partidos de la
jornada 1 a la N ya tienen resultado real en `epl_matches.csv` -cruzando contra el
calendario, igual que `season_trajectory.py`- y ajustando el `--as-of` a esa fecha. Es
la predicción completa, no sólo el resumen de título: clasificación, partidos, matriz
de posiciones, todo anclado a esa jornada.

Cada corrida con `--through N` archiva sus ficheros con el sufijo `_JN`
(`season_predictions_J5.xlsx`, `season_probabilities_J5.csv`, etc.) en vez de pisar
siempre los mismos nombres — así se acumula un Excel por jornada en `results/` y se
puede comparar cómo cambió la predicción de una jornada a la siguiente.

```bash
python scripts/simulate_season.py --through 5
```

Incompatible con `--played` -- si pasas los dos, aborta con un error en vez de elegir
uno en silencio. Mientras `epl_matches.csv` no tenga partidos reales de esa jornada,
se comporta igual que sin `--through` (proyección de pretemporada), avisándolo por log.

El CSV debe tener las columnas `home`, `away`, `home_goals`, `away_goals`, con los
nombres en nomenclatura canónica (la de `team_names.csv`).

Ejemplo del efecto — Arsenal perdiendo sus 5 primeros partidos:

| | Sin jugar | Tras 5 derrotas |
|---|---|---|
| Arsenal, título | 49,1% | 12,1% |
| Arsenal, top-4 | 96,9% | 71,3% |
| Man City, título | 42,8% | 67,5% |

### Paso 5 — Predicciones individuales

```bash
python scripts/predict_players.py
```

Necesita `data/raw/appearances.csv` (ver Fuentes de datos). Reparte los goles de cada
simulación de equipo entre los jugadores de su plantilla según su cuota histórica de
goles/asistencias, y da la probabilidad de Bota de Oro y máximo asistente de cada
jugador — no un modelo de goles por jugador independiente, ver la nota metodológica
más abajo.

| Opción | Efecto |
|---|---|
| `--sims 50000` | Número de temporadas simuladas (igual que en `simulate_season.py`) |
| `--played fichero.csv` | Incorporar resultados ya disputados |
| `--top 30` | Filas a mostrar de cada tabla |

### Paso 6 — Cómo evolucionan las probabilidades en la temporada

```bash
python scripts/season_trajectory.py
```

Simula la temporada tal como se veía **en cada jornada**, desde la 0 (pretemporada)
hasta la última con resultados reales cargados: para la jornada N usa como partidos ya
jugados sólo los que de verdad se disputaron hasta ahí (detectados automáticamente
cruzando el calendario con `epl_matches.csv`, sin mantener ningún CSV a mano) y simula
el resto. Para la predicción COMPLETA de una única jornada -no sólo el título, también
partidos y matriz de posiciones- usa `simulate_season.py --through N` (Paso 4).

Guarda todas las jornadas en `results/title_trajectory.csv` (formato largo, una fila
por equipo y jornada) y en `results/title_trajectory.xlsx`, con una hoja adicional por
cada métrica (`P(título)`, `P(top-4)`, `P(descenso)`, `Puntos medios`) ya pivotada
jornada × equipo — selecciona el rango en Excel e inserta un gráfico de líneas
directamente, sin tener que armar la tabla dinámica.

Mientras `epl_matches.csv` no tenga partidos de 2026/27 (ver Paso 2), todas las
jornadas devuelven la misma proyección de pretemporada — no es un error, es que
todavía no hay ningún resultado real que incorporar.

| Opción | Efecto |
|---|---|
| `--to 10` | Sólo hasta la jornada 10 |
| `--sims 5000` | Simulaciones por jornada (default ya es 5.000, para que 39 jornadas no tarden minutos) |

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
| `player_predictions.csv` | ~785 | Uno por jugador con historial en Premier: goles/asistencias medios, P10/P90, P(Bota de Oro), P(máximo asistente). Genera `predict_players.py` |
| `title_trajectory.csv` | 20 × jornadas | Probabilidades de título/top-4/descenso de cada equipo, una vez por jornada simulada. Genera `season_trajectory.py` |
| `season_predictions.xlsx` | — | `season_probabilities`, `match_predictions`, `position_matrix` y `points_distribution`, una por hoja. Mismo contenido que los CSV de arriba, junto en un Excel |
| `title_trajectory.xlsx` | — | El detalle de `title_trajectory.csv` más una hoja por métrica ya pivotada jornada × equipo, lista para graficar |

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

Bota de Oro y máximo asistente, a partir de una simulación ya hecha:

```python
from src.players import load_appearances, player_shares, simulate_player_stats

appearances = load_appearances()
shares = player_shares(appearances, as_of="2026-08-21", teams=list(teams["canonical"]))
jugadores = simulate_player_stats(res, shares)

print(jugadores.sort_values("p_bota_oro", ascending=False).head(10))
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
│   │   ├── appearances.csv                   # goles/asistencias por partido — no en git, ver abajo
│   │   └── competitions.csv                  # códigos de competición
│   └── processed/                            # generado — en .gitignore
├── src/                                      # librería
│   ├── __init__.py
│   ├── utils.py                              # rutas, carga y normalización
│   ├── ratings.py                            # motor Dixon-Coles
│   ├── squad.py                              # valor de plantilla (Transfermarkt)
│   ├── congestion.py                         # descanso entre partidos, todas las competiciones
│   ├── players.py                            # reparto de goles/asistencias por jugador
│   └── simulator.py                          # Monte Carlo y exportación
├── scripts/                                  # ejecutables
│   ├── download_footballdata.py              # ingesta reproducible
│   ├── backtest_ratings.py                   # validación del motor
│   ├── calibrate_squad_prior.py              # calibración del prior de plantilla
│   ├── calibrate_congestion.py               # calibración (descartada) de la fatiga
│   ├── calibrate_players.py                  # calibración del reparto de goles
│   ├── simulate_season.py                    # predicción de la temporada
│   ├── predict_players.py                    # Bota de Oro y máximo asistente
│   ├── update_current_season.py              # refresco directo de football-data.co.uk
│   └── season_trajectory.py                  # probabilidades jornada a jornada
├── .github/workflows/
│   └── update_results.yml                    # cron diario para update_current_season.py
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
| `ratings.py` | **Modelado.** `fit_ratings()` ajusta ataque, defensa y ventaja de local; `apply_defaults()` corrige equipos sin historia; `apply_squad_prior()` mezcla el rating con el valor de plantilla; `predict_match()` da probabilidades 1X2 |
| `squad.py` | **Plantillas.** `squad_value_at()` valor de plantilla histórico (player_valuations.csv); `current_squad_value()` valor actual, post-fichajes (players.csv) |
| `congestion.py` | **Calendario.** `rest_days()`/`matches_in_window()` descanso y carga reciente de un equipo, contando todas las competiciones de club (games.csv) |
| `players.py` | **Jugadores.** `player_shares()` cuota histórica de goles/asistencias de cada jugador sobre su equipo; `simulate_player_stats()` reparte los goles de cada simulación de equipo entre su plantilla |
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

### Proyección 2026/27 (ratings históricos + valor de plantilla)

```
  #   Equipo              Pts    P10-P90   Título    Top-4    Desc.
  1   Arsenal            76.1      66-86    46.3%    94.6%     0.0%
  2   Man City           75.5      66-85    42.3%    93.8%     0.0%
  3   Liverpool          64.5      54-75     5.3%    55.2%     0.1%
  4   Man United         59.8      49-70     1.7%    31.3%     0.3%
  5   Chelsea            57.8      47-68     1.0%    22.8%     0.6%
  6   Aston Villa        56.6      46-67     0.7%    18.2%     0.8%
  7   Newcastle          56.4      46-67     0.8%    17.4%     0.9%
  ...
  18  Coventry           34.1      25-43     0.0%     0.0%    72.2%
  19  Ipswich            31.9      23-41     0.0%     0.0%    81.4%
  20  Hull               30.2      22-39     0.0%     0.0%    87.6%
```

Frente a los ratings puramente históricos, Chelsea sube del 7º al 5º puesto (top-4
17.0% → 22.8%: tiene la 3ª plantilla más valiosa de la liga) y Hull cae más
(descenso 73.7% → 87.6%: la plantilla más barata de las 20). Movimientos reales
pero acotados — es la firma de un prior con peso 0,15, no de un modelo que haya
sustituido la historia por el mercado.

> **Congestión de calendario: probada y descartada.** El valor de plantilla
> ya está incorporado; la fatiga por calendario apretado (Champions/Europa,
> copas domésticas) se midió y no mostró efecto detectable en el backtest —
> ver la nota metodológica más abajo para los números.

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

### Congestión de calendario: descartada

Jugar Champions League el martes y Premier el sábado deja menos descanso que una
semana sin competición europea — la hipótesis, y el último bloque pendiente del
roadmap, era que esa fatiga debía notarse en los goles. `scripts/calibrate_congestion.py`
la puso a prueba de dos formas distintas, contando siempre TODAS las competiciones de
club vía `games.csv` (liga, FA Cup, Carabao, UEFA), no sólo Premier:

| | Cobertura | coef (MLE) | log-loss | RPS |
|---|---|---|---|---|
| Sin ajuste | — | — | 0,9755 | 0,2004 |
| A. Déficit de descanso | 43,0% de locales con descanso corto | +0,0027 | 0,9755 | 0,2004 |
| B. Carga reciente (10 días) | 5,7% de locales con carga alta | +0,0244 | 0,9755 | 0,2004 |

**Ninguna mejora el backtest.** El coeficiente de máxima verosimilitud sale casi cero
y con el signo equivocado en las dos — el esperado era negativo (menos descanso, menos
goles). RPS y log-loss no se mueven ni en la cuarta cifra decimal. Es un resultado
real, no ruido: 4.180 partidos evaluados, la misma metodología jornada a jornada que
calibró la vida media y el prior de plantilla.

¿Por qué no aparece el efecto? Con equipos que rotan plantilla en copas y una liga de
20 equipos donde casi todos —no sólo los europeos— tienen tramos de calendario
apretado (Navidad, repesca de copa), el "descanso" deja de distinguir bien a un
equipo de otro. La fatiga puede ser real a nivel de jugador sin serlo a nivel de
resultado de equipo, que es la única unidad que este modelo observa.

**El mecanismo queda implementado, no la conclusión forzada.** `CONGESTION_COEF = 0.0`
en `src/ratings.py` lo deja inerte por defecto; `fatigue_multiplier()`,
`src/congestion.py` y el flag `--no-congestion` están listos para recalibrar si esto
cambia con más temporadas o una medida de carga distinta a las dos probadas aquí.
Además, `games.csv` se corta en marzo de 2026 (ver limitación de `player_valuations.csv`
más abajo), así que hoy no aportaría nada a la temporada 2026/27 aunque el efecto
existiera: no hay calendario europeo futuro con el que calcular el descanso.

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

### Prior de valor de plantilla

Los ratings de Dixon-Coles son puramente de resultados: no saben nada de los
fichajes de un verano hasta que el equipo lleva partidos jugados con la
plantilla nueva. `src/squad.py` añade una segunda señal — el valor de mercado
agregado de la plantilla (Transfermarkt) — y `apply_squad_prior()` la mezcla
con el rating histórico.

**Medido, no inventado**, igual que los defaults de ascendidos.
`scripts/calibrate_squad_prior.py` responde dos preguntas por separado:

1. **¿Cómo se traduce el valor de plantilla en rating?** Regresión de
   log(ataque) y log(defensa) contra el valor de plantilla relativo a la
   media de la liga esa temporada, sobre 220 equipo-temporada (2015/16 –
   2025/26):

   | | Pendiente | Intercepto | R² |
   |---|---|---|---|
   | Ataque  | +0,2842 | +0,0258 | 0,505 |
   | Defensa | −0,2056 | −0,0740 | 0,393 |

   El signo es el esperado en ambos: más valor de plantilla → más ataque y
   mejor defensa (recuerda que en este modelo defensa < 1 es bueno). La
   plantilla explica una parte real del rendimiento — no toda: R² de 0,4-0,5,
   no 0,9.

2. **¿Cuánto debe pesar frente al rating histórico?** Barrido del peso de
   mezcla en el mismo backtest jornada a jornada:

   | Peso | RPS | | Peso | RPS |
   |---|---|---|---|---|
   | 0,00 | 0,2004 | | 0,20 | 0,2003 |
   | 0,05 | 0,2003 | | 0,25 | 0,2004 |
   | 0,10 | 0,2003 | | 0,30 | 0,2005 |
   | **0,15** | **0,2002** | | 0,40 | 0,2009 |

   Curva en U con pico en 0,10-0,15: mejora real pero modesta (log-loss
   0,9755 → 0,9751), justo lo que cabe esperar de una señal que ya está
   parcialmente contenida en la historia de un equipo con plantilla estable.

**Dos fuentes distintas según la fecha.** `player_valuations.csv` (la serie
histórica de valoraciones) se corta en marzo de 2026, antes de que cerrara el
mercado de verano el 31 de agosto. Sirve para calibrar sobre temporadas
pasadas (`squad_value_at`), pero no para la temporada 2026/27 en sí. Para eso
se usa `current_squad_value`, que lee el club y valor **actuales** de cada
jugador en `players.csv` — el único fichero del repositorio que refleja los
fichajes ya cerrados.

**A diferencia de los defaults de ascendidos**, que sólo corrigen a equipos
sin historia, el prior de plantilla se aplica a los 20 equipos con dato de
Transfermarkt: un equipo con mucha historia pero una plantilla muy reforzada
este verano también debe moverse. Coventry, sin `tm_club_id` en
`team_names.csv`, queda fuera y se apoya sólo en el default de ascendido.

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

### Predicciones individuales: reparto, no un modelo de jugador aparte

`src/players.py` no predice goles de un jugador de forma independiente -exigiría
minutos y titularidades futuras, que no están en los datos-. En vez de eso REPARTE lo
que `simulate_season()` ya genera: para cada una de las n_sims temporadas simuladas,
toma los goles totales del equipo (`res.goals_for`) y los sortea entre su plantilla con
un multinomial ponderado por la cuota histórica de cada jugador
(`goles_del_jugador / goles_del_equipo`, ponderada por recencia). Las asistencias salen
igual, escalando los goles de equipo por `ASSISTS_PER_GOAL = 0,792` (asistencias/goles
medido sobre las 147.550 apariciones de Premier de `appearances.csv`, 2012/13-2025/26).

**Validado con una pregunta de ranking, no de probabilidad.** Los otros calibradores de
este proyecto usan log-loss/RPS porque predicen la probabilidad de un resultado
concreto; aquí la pregunta es "¿quién marca más?", así que la métrica es acierto simple
(hit-rate): tomando sólo información anterior a cada temporada (2016/17-2025/26),
¿el jugador con más cuota histórica de goles de su equipo es su máximo goleador real
esa temporada? `scripts/calibrate_players.py` barre la vida media del peso temporal:

| Vida media | Acierto | | Vida media | Acierto |
|---|---|---|---|---|
| 180 d | 34,2% | | 545 d | 34,2% |
| 270 d | **36,3%** | | 730 d | 33,7% |
| 365 d | 34,7% | | | |

Sobre 190 equipo-temporada, sin ningún pico claro (33,7%-36,3%): a diferencia de la
vida media de `ratings.py`, aquí la elección de vida media importa poco. **Referencias:**
elegir al azar entre una plantilla de ~20 con minutos relevantes acertaría el 5,0%; el
modelo, con 270 días, el 36,3% — siete veces mejor que el azar, una señal real. Pero
**repetir sin más el goleador de la temporada anterior acierta el 36,6%**, prácticamente
empatado con el modelo en esta métrica concreta.

Eso no vacía de sentido el reparto por cuota: a diferencia de "repetir el goleador
anterior" -que no tiene respuesta cuando ese jugador se ha ido o el equipo acaba de
ascender (cobertura de sólo 153 equipo-temporada frente a 190)-, el modelo cubre
cualquier plantilla con historial, se ajusta automáticamente a fichajes y da una
distribución de probabilidad completa por jugador, no una única apuesta. Pero para la
pregunta estrecha de "quién es el número 1", esta calibración no encontró ventaja clara
sobre la referencia más simple posible, y así se reporta.

**Limitación conocida:** la cuota de cada jugador es fija dentro de las n_sims
simulaciones -sólo varía cuánto marca el equipo, no cómo se reparte dentro de él-, así
que las probabilidades de Bota de Oro están algo más ajustadas de lo que sería una
predicción con toda la incertidumbre real (lesiones, rotación, un fichaje de invierno)
incorporada.

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
| `games.csv`, `players.csv`, `transfers.csv`, `player_valuations.csv`, `appearances.csv` | [Transfermarkt / Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores) | Partidos de club, plantillas, valores de mercado, goles/asistencias por partido |
| `premier-league-gb-eng_2026-27.csv` | Calendario oficial | 380 fixtures de la temporada |

`appearances.csv` (140 MB, 1,86M apariciones) no está en el repositorio por tamaño —
sigue en `.gitignore`, hace falta para `src/players.py` y no viene incluido al clonar.
Descárgalo del mismo dataset de Kaggle y colócalo en `data/raw/appearances.csv`. Si
Kaggle no es accesible desde donde ejecutes esto, una alternativa es subirlo como
adjunto binario a un [Release de GitHub](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
(hasta 2GB por archivo, sin las limitaciones de tamaño de git) y descargarlo desde ahí.

---

## Licencia

MIT
