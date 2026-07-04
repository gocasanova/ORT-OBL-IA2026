# MountainCarContinuous con Q-Learning y Dyna-Q

Esta carpeta contiene agentes tabulares Q-Learning y Dyna-Q para ejecutar y
comparar experimentos de `MountainCarContinuous-v0`. La notebook existente se
conserva para Q-Learning; Dyna-Q se ejecuta con los scripts documentados abajo.
El entrenamiento y el test no abren ventanas.

## Instalación y ejecución

Desde esta carpeta:

```bash
poetry install
poetry run jupyter notebook continuous_mountain_car.ipynb
```

La notebook tiene un flujo único y breve: imports, función, configuración
editable, ejecución y gráficos. La celda titulada
`▶ EJECUTÁ ESTA CELDA PARA ENTRENAR` es la única que inicia el entrenamiento.
Las búsquedas múltiples continúan disponibles desde la consola.

## Hiperparámetros admitidos

Los experimentos nuevos utilizan únicamente `alpha`, `gamma`, `epsilon`,
`epsilon_min`, `epsilon_decay`, cantidad de episodios y bins de posición,
velocidad y acciones. La Q-table comienza en cero, el reward de aprendizaje es
el reward original del ambiente y las acciones siempre se distribuyen
uniformemente entre `-1` y `1`.

Los modelos antiguos con configuraciones adicionales siguen pudiéndose cargar,
pero esos campos ya no forman parte del entrenamiento ni de las búsquedas.

## Búsqueda de hiperparámetros

`hyperparameter_search.py` automatiza la ejecución y el registro, pero no evalúa
cuál configuración es preferible. Antes de una búsqueda conviene inspeccionar
las tareas sin entrenar ni modificar archivos:

```bash
poetry run python hyperparameter_search.py --mode manual --dry-run
```

Para validar dos configuraciones manuales rápidamente:

```bash
poetry run python hyperparameter_search.py \
  --mode manual --episodes 100 --max-steps 300 \
  --test-episodes 2 --seeds 42 --limit 2
```

Para el entrenamiento manual completo:

```bash
poetry run python hyperparameter_search.py \
  --mode manual --episodes 10000 --test-episodes 20 --seeds 42
```

Se muestra progreso cada 500 episodios. Puede cambiarse con
`--progress-interval 1000` o desactivarse con `--quiet`.

Al principio del script están `SEARCH_NAME`, `SEARCH_MODE`, `EPISODES`,
`MAX_STEPS`, `TEST_EPISODES`, `SEEDS` y `DRY_RUN`. El modo `grid` combina los
valores de las listas del grid; el modo `manual` recorre `MANUAL_CONFIGS`. También
se pueden sobrescribir parámetros desde la consola, por ejemplo:

```bash
poetry run python hyperparameter_search.py \
  --mode manual --episodes 3000 --test-episodes 20 --seeds 42 123
```

El grid predeterminado contiene ocho configuraciones por semilla. La opción
`--limit` sirve para pruebas controladas sin modificar la definición del grid.
Cada corrida terminada se registra inmediatamente. Si hay un error o
interrupción, las anteriores se conservan y la búsqueda queda marcada como
`failed` o `interrupted`.

## Perfil overnight

El perfil `overnight` combina 12 configuraciones con 5 seeds (`42`, `123`,
`999`, `2026` y `777`): son 60 entrenamientos de 20.000 episodios y 100 tests
greedy cada uno. Varía solamente `alpha`, `gamma`, `epsilon`, `epsilon_min`,
`epsilon_decay` y la cantidad de bins de posición, velocidad y acciones.

Antes de iniciarlo se puede revisar el plan y una estimación basada en tiempos
históricos, sin escribir archivos:

```bash
poetry run python hyperparameter_search.py --profile overnight --dry-run
```

La ejecución recomendada permite continuar después de un corte:

```bash
poetry run python hyperparameter_search.py --profile overnight --resume
```

Cada combinación posee un `run_key` estable. `--resume` omite una corrida solo
si ese `run_key` ya está registrado y su modelo existe. Para repetir todo aunque
existan resultados se usa:

```bash
poetry run python hyperparameter_search.py --profile overnight --rerun
```

Validación corta del perfil:

```bash
poetry run python hyperparameter_search.py --profile overnight \
  --episodes 500 --test-episodes 5 --seeds 42 --limit 2
```

El progreso informa tiempo acumulado y ETA después de cada experimento. Los
resultados agregados se regeneran en
`results/q_learning_overnight_summary.csv`, agrupados por `search_id`, perfil y
configuración. Sus medias y desvíos son descriptivos: no se ordenan, puntúan ni
seleccionan modelos.

## Registrar un experimento

Cada configuración debe incluir `config_name`, los hiperparámetros y la cantidad
de episodios. Luego se ejecuta:

```python
agent, history, test_results, result_row = run_experiment(
    config,
    seed=42,
    notes="Descripción de la prueba",
)
```

La función entrena, testea sin exploración, guarda un modelo con nombre único en
`models/` y agrega una fila a:

```text
results/q_learning_experiments.csv
```

Las filas existentes nunca se reemplazan. `experiment_logger.py` mantiene un
esquema fijo, escribe el encabezado una sola vez y permite cargar los resultados
con `load_experiment_results()`.

Cada modelo queda asociado a su configuración, semilla e identificador:

```text
models/q_learning_<config_name>_seed<seed>_<experiment_id>.pkl
```

## Métricas registradas

- Identificación: `experiment_id`, `timestamp`, `search_id`, `search_name`,
  `config_index`, `total_configs`, `config_name`, `seed` y `notes`.
- Configuración: bins, `alpha`, `gamma`, epsilon inicial/final,
  `epsilon_decay`, `episodes` y `max_steps`.
- Entrenamiento: recompensa original y de aprendizaje, posición media/máxima,
  pasos y porcentaje de éxito de los últimos 100 episodios.
- Test: cantidad/porcentaje de éxitos, recompensa original, pasos y máxima
  posición media.
- Ejecución: `training_time_seconds` y `model_path`.

Las tasas de éxito se expresan como porcentajes entre 0 y 100. El tiempo incluye
solamente el entrenamiento, no el test ni el guardado.

Los experimentos quedan en `results/q_learning_experiments.csv`. Cada búsqueda
agrega además una fila a `results/q_learning_search_runs.csv`, con sus timestamps,
tiempo total, cantidad planificada/completada, semillas y estado. Las corridas
hechas directamente desde la notebook dejan vacíos los campos de búsqueda.

## Experimento desde la notebook

Cada ejecución corresponde a una sola configuración y muestra sus métricas de
test, tiempo y modelo guardado. Los gráficos presentan recompensa, promedio
móvil y máxima posición por episodio. Una posición cercana a `0.45` indica que
el auto está aproximándose a la meta.

## Discretización

La observación `[posición, velocidad]` se divide en una grilla cuyos límites se
leen de `env.observation_space`. La fuerza se discretiza con valores uniformes
entre `-1` y `1`. La tabla Q tiene forma:

```text
position_bins × velocity_bins × action_bins
```

Antes de llamar a `env.step`, cada acción se convierte en un array NumPy de forma
`(1,)` y tipo `float32`.

## Archivos principales

- `q_learning_agent.py`: agente, entrenamiento, test y persistencia.
- `experiment_logger.py`: resúmenes y almacenamiento acumulativo en CSV.
- `hyperparameter_search.py`: búsqueda grid o manual y manejo de interrupciones.
- `continuous_mountain_car.ipynb`: experimento editable y gráficos de aprendizaje.
- `results/q_learning_experiments.csv`: historial comparable de corridas.
- `results/q_learning_search_runs.csv`: duración y estado de cada búsqueda.
- `results/q_learning_overnight_summary.csv`: agregados por búsqueda y
  configuración sobre las seeds completadas.
- `models/*.pkl`: modelos producidos por cada experimento.

Los archivos `.pkl` solo deben cargarse si provienen de una fuente confiable.

## Dyna-Q

`DynaQAgent` reutiliza la discretización, las acciones y la evaluación de
`QLearningAgent`, pero inicia una Q-table propia y mantiene un modelo tabular del
ambiente. Nunca carga un `.pkl` de Q-Learning para entrenar. La configuración
base replica `baseline_40x40_a11`, definida para Q-Learning, y el
baseline se usa únicamente al generar la comparación.

Entrenamiento completo con los cinco valores pedidos:

```bash
poetry run python scripts/train_dyna_q.py \
  --planning-steps 0 5 10 20 50 \
  --episodes 20000 --evaluation-episodes 100
```

Para comparar estabilidad con las mismas cinco seeds del overnight de
Q-Learning:

```bash
poetry run python scripts/train_dyna_q.py \
  --planning-steps 0 5 10 20 50 \
  --episodes 20000 --evaluation-episodes 100 \
  --seeds 42 123 999 2026 777
```

Cada corrida guarda un modelo individual y actualiza
`models/dyna_q_best.pkl` con el mejor del lote. Los CSV son
`results/dyna_q_training_results.csv` (una fila por episodio) y
`results/dyna_q_evaluation_results.csv` (una fila por evaluación).

Para cargar y reevaluar el mejor modelo sin exploración:

```bash
poetry run python scripts/evaluate_dyna_q.py \
  --model models/dyna_q_best.pkl --episodes 100
```

Para crear `results/comparison_qlearning_dynaq.csv` y los PNG en
`results/plots/`, usando únicamente datos registrados:

```bash
poetry run python scripts/generate_dyna_q_reports.py
```

El generador usa el `run_id` Dyna-Q más reciente. Puede elegirse otro con
`--run-id ID`. La comparación de Q-Learning toma las corridas más extensas de
`baseline_40x40_a11` disponibles en
`results/q_learning_experiments.csv`; si faltan, primero se debe ejecutar el
perfil overnight documentado arriba.

Validación corta del flujo, útil antes de una corrida larga:

```bash
poetry run python scripts/train_dyna_q.py \
  --planning-steps 0 5 --episodes 5 --max-steps 100 \
  --evaluation-episodes 2 --seeds 42
```

La base para redactar resultados y los placeholders pendientes están en
`docs/dyna_q_resultados.md`.
