
# Isolation

Breve descripción de los archivos contenidos en este directorio.

- **agent.py**: Clase abstracta `Agent` que define la interfaz de los agentes.
- **board.py**: Implementación de la clase `Board` con la representación del tablero y la lógica del juego.
- **input_agent.py**: `InputAgent` sencillo para jugar manualmente desde la consola.
- **random_agent.py**: `RandomAgent` que elige una acción legal al azar usando `board.get_possible_actions()`.
- **stratagem.py**: Agente con una implementación ofuscada.
- **isolation_env.py**: Wrapper Gym `IsolationEnv` que adapta `Board` a la API.
- **play.py**: Utilidad `play_vs_other_agent` para ejecutar partidas entre dos agentes y opcionalmente mostrar el tablero en cada turno.
- **isolation.ipynb**: Notebook Jupyter con demostraciones y ejemplos interactivos del entorno y agentes.
- **mate_isolation_experiments.ipynb**: Evidencia reproducible de MATE con validaciones, tablas y gráficos para el informe.

## Parte 2: MATE

La implementación de búsqueda está en `mate_agents.py` e incluye:

- `MinimaxAgent`
- `AlphaBetaAgent` (con métricas de poda)
- `ExpectimaxAgent` (rival uniforme)

Las heurísticas seleccionables están en `mate_evaluations.py`: `mobility`,
`aggressive`, `defensive`, `balanced`, `territory` y `weighted`.

### Instalación y pruebas

Desde este directorio:

```bash
poetry install
poetry run python -m unittest discover -s tests -v
```

Desde la raíz general del repositorio, el equivalente es:

```bash
poetry -C Isolation install
poetry -C Isolation run python -m unittest discover -s tests -v
```

### Experimentos

Validación mínima y rápida:

```bash
poetry run python experiments_mate.py --suite smoke --games 1 --depths 1
```

Comparación principal para obtener resultados del informe:

```bash
poetry run python experiments_mate.py --suite core --games 20 --depths 1 2 3
```

Sin argumentos se ejecuta una versión corta del modo `core` (dos partidas por
configuración y profundidades 1 y 2):

```bash
poetry run python experiments_mate.py
```

Comparación completa, incluido el todos-contra-todos de heurísticas:

```bash
poetry run python experiments_mate.py --suite full --games 20 --depths 1 2 3
```

Profundidades grandes, especialmente con Minimax o Expectimax, pueden demorar
por el alto factor de ramificación. Para una prueba corta conviene usar
profundidades 1 y 2.

Los resultados se sobrescriben en:

- `results/mate_experiments.csv`
- `results/mate_experiments.json`

El resumen reutilizable para el informe está en `docs/mate_summary.md`.

El notebook `mate_isolation_experiments.ipynb` carga el CSV anterior y guarda
cuatro gráficos PNG en `results/`. Después de `poetry install`, puede abrirse en
VS Code o Jupyter seleccionando el entorno virtual de Poetry como kernel.
