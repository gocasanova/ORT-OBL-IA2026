# MountainCarContinuous con Q-Learning

Esta carpeta contiene un agente Q-Learning tabular y una notebook para ejecutar
y comparar experimentos de `MountainCarContinuous-v0`. El entrenamiento y el
test usan `render_mode="rgb_array"` y no abren ventanas.

## Instalación y ejecución

Desde esta carpeta:

```bash
poetry install
poetry run jupyter notebook continuous_mountain_car.ipynb
```

La notebook incluye configuraciones breves para experimentación interactiva. Las
búsquedas largas se ejecutan desde la consola para conservar el progreso aunque
se cierre Jupyter.

Para comparar resultados en la notebook:

1. Ejecutar las celdas; los ejemplos no entrenan mientras
   `RUN_EXAMPLE_EXPERIMENTS=False`.
2. Editar `COMPARISON_SEARCH_ID` para elegir una búsqueda. La vista inicial usa
   la overnight completa `6b358d688579`.
3. Revisar la tabla y las seis gráficas agregadas por configuración/seed.
4. Escribir manualmente un nombre en `CHOSEN_CONFIG_NAME` para ver sus cinco
   corridas y rutas de modelos.

La notebook no asigna puntajes ni decide una configuración.

## Estrategias contra la política no-op

Las primeras búsquedas aprendieron a usar fuerza cero: así evitaban la
penalización energética, pero nunca alcanzaban la meta. El agente permite ahora:

- `reward_shaping="potential"`: durante entrenamiento agrega
  `gamma * Phi(next_state) - Phi(state)` para aportar señal por posición y
  velocidad. El test siempre usa y reporta la recompensa original del ambiente.
- `q_init > 0`: inicialización optimista que incentiva probar acciones no
  visitadas.
- `explicit_action_values`: lista opcional que sustituye las acciones uniformes;
  puede omitirse `0.0` para comparar políticas sin no-op.
- decay lento de epsilon, por ejemplo `0.9995` o `0.9997`.

`reward_shaping="none"` y `q_init=0.0` conservan el comportamiento base.

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
greedy cada uno. Incluye variaciones de `q_init`, cantidad de acciones, shaping,
gamma, alpha y listas sin acción cero.

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
- Configuración: bins, `alpha`, `gamma`, epsilon inicial/final, shaping,
  `q_init`, acciones explícitas, `episodes` y `max_steps`.
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

## Comparación manual

La notebook muestra el CSV completo, una tabla reducida y cuatro gráficos:

- tasa de éxito de test;
- recompensa media de test;
- pasos medios de test;
- tiempo de entrenamiento.

No se ordenan resultados ni se selecciona o reentrena automáticamente una
configuración. Cada llamada a `run_experiment` corresponde a una única corrida.
La elección queda a criterio del usuario, considerando especialmente
`test_success_rate`, `test_success_count`, `test_avg_reward`,
`test_avg_max_position`, `train_last_100_max_position`, estabilidad, tiempo y
complejidad de la grilla. Una máxima posición cercana a `0.45` indica que el auto
está aproximándose a la meta aunque todavía no la alcance consistentemente.

## Discretización

La observación `[posición, velocidad]` se divide en una grilla cuyos límites se
leen de `env.observation_space`. La fuerza se discretiza con valores uniformes
entre `-1` y `1`, salvo que se proporcionen acciones explícitas. La tabla Q tiene
forma:

```text
position_bins × velocity_bins × action_bins
```

Antes de llamar a `env.step`, cada acción se convierte en un array NumPy de forma
`(1,)` y tipo `float32`.

## Archivos principales

- `q_learning_agent.py`: agente, entrenamiento, test y persistencia.
- `experiment_logger.py`: resúmenes y almacenamiento acumulativo en CSV.
- `hyperparameter_search.py`: búsqueda grid o manual y manejo de interrupciones.
- `continuous_mountain_car.ipynb`: ejecución, tablas y gráficos descriptivos.
- `results/q_learning_experiments.csv`: historial comparable de corridas.
- `results/q_learning_search_runs.csv`: duración y estado de cada búsqueda.
- `results/q_learning_overnight_summary.csv`: agregados por búsqueda y
  configuración sobre las seeds completadas.
- `models/*.pkl`: modelos producidos por cada experimento.

Los archivos `.pkl` solo deben cargarse si provienen de una fuente confiable.
