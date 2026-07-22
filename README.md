# Premier League 2026/27 Predictor

Predicción de la temporada 2026/27 de la Premier League mediante un modelo de goles
Dixon-Coles con ratings dinámicos y simulación Monte Carlo de la temporada completa.

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

La consecuencia práctica: donde el Mundial necesitaba fuerza estática más simulación
de bracket, aquí hace falta **fuerza dinámica en el tiempo** y **simulación de
temporada completa**.

---

## Estado del proyecto

- [x] Ingesta de resultados históricos (2005/06 – 2025/26)
- [x] Unificación de nomenclaturas entre fuentes
- [x] Definición de los 20 equipos de 2026/27
- [ ] Motor de ratings ataque/defensa vectorizado
- [ ] Features de plantilla, fichajes y congestión de calendario
- [ ] Modelo de goles esperados
- [ ] Simulador de liga (Monte Carlo)
- [ ] Backtest sobre temporadas pasadas

---

## Estructura

```
premier-league-predictor/
├── data/
│   ├── raw/
│   │   ├── epl_matches.csv          # 7.980 partidos, 21 temporadas
│   │   ├── team_names.csv           # mapeo football-data ↔ Transfermarkt
│   │   ├── teams_2026_27.csv        # los 20 equipos de la temporada
│   │   ├── games.csv                # Transfermarkt: PL + Europa + copas
│   │   ├── player_valuations.csv    # valores de mercado con histórico
│   │   ├── transfers.csv            # movimientos de fichajes
│   │   ├── players.csv              # metadatos de jugadores
│   │   └── competitions.csv         # códigos de competición
│   └── processed/                   # generado — en .gitignore
├── scripts/
│   └── download_footballdata.py     # ingesta reproducible
├── src/
├── models/                          # en .gitignore
├── results/                         # en .gitignore
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
git clone https://github.com/<usuario>/premier-league-predictor.git
cd premier-league-predictor
pip install -r requirements.txt
```

### Actualizar los resultados

El fichero `epl_matches.csv` se regenera con:

```bash
python scripts/download_footballdata.py
```

Descarga desde [datasets/football-datasets](https://github.com/datasets/football-datasets),
un espejo de [football-data.co.uk](https://www.football-data.co.uk/) que se actualiza
a diario. Durante la temporada 2026/27 basta con reejecutarlo para incorporar las
jornadas nuevas y reentrenar.

---

## Fuentes de datos

| Dataset | Fuente | Contenido |
|---|---|---|
| `epl_matches.csv` | football-data.co.uk vía datahub | Resultados, tiros, córners, tarjetas |
| `games.csv`, `players.csv`, `transfers.csv`, `player_valuations.csv` | [Transfermarkt / Kaggle](https://www.kaggle.com/datasets/davidcariboo/player-scores) | Partidos de club, plantillas, valores de mercado |

`appearances.csv` (140 MB) no está en el repositorio. Descárgalo del mismo dataset de
Kaggle si vas a usar el módulo de predicciones individuales.

---

## Temporada 2026/27

Arranca el **22 de agosto de 2026** y termina el **30 de mayo de 2027**: 33 jornadas
de fin de semana y 5 entre semana. El inicio y la última jornada se retrasaron una
semana por el Mundial 2026. La ventana de fichajes de verano va del 15 de junio al
31 de agosto de 2026.

**Descendidos de 2025/26:** Wolves (20º), Burnley (19º), West Ham (18º)
**Ascendidos del Championship:** Coventry City (campeón), Ipswich Town (2º), Hull City (playoff)

Arsenal defiende el título.

---

## Notas metodológicas

**Nomenclatura de equipos.** football-data usa nombres cortos (`Man City`) y
Transfermarkt nombres largos (`Manchester City Football Club`). De 44 equipos, solo
uno coincidía literalmente. `team_names.csv` resuelve el mapeo; cruzar las fuentes sin
él devuelve prácticamente cero filas.

**Equipos ascendidos.** No tienen histórico reciente en Primera, así que el modelo les
asignaría rating medio y los situaría a mitad de tabla. Se corrige con un prior basado
en el comportamiento histórico de los ascendidos, ajustado por valor de plantilla.
Coventry City además no aparece en los ficheros de Transfermarkt del proyecto, porque
llevaba 25 años fuera de la máxima categoría.

**Fechas.** Los CSV de football-data vienen en ISO (`YYYY-MM-DD`), no en formato
británico. El script de ingesta fuerza el formato explícito y aborta si algo no parsea:
si se deja inferir, pandas invierte día y mes en silencio y ensucia el decaimiento
temporal sin dar ningún error.

---

## Licencia

MIT
