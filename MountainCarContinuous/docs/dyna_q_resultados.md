# Dyna-Q en MountainCarContinuous-v0

## Implementación

Dyna-Q fue implementado como una extensión de Q-Learning tabular. Hereda la
misma discretización de observaciones, selección epsilon-greedy, grilla de
acciones y evaluación greedy de `QLearningAgent`, pero crea una Q-table nueva y
no carga ningún modelo `.pkl` de Q-Learning.

La configuración base coincide con `baseline_40x40_a11` de Q-Learning: 40 bins
de posición, 40 de velocidad, 11 acciones uniformes, alpha 0,1, gamma 0,995,
epsilon inicial 1, epsilon mínimo 0,1 y decay 0,9995. Ambas Q-tables comienzan en
cero y usan el reward original. El modelo interno guarda la última transición
observada para cada par discreto:

```text
model[(state, action)] = (reward, next_state, done)
```

`done` vale verdadero cuando Gymnasium informa `terminated` o `truncated`. Luego
de cada paso real se muestrean al azar `planning_steps` transiciones del modelo y
se vuelve a aplicar la actualización de Q-Learning. Con `planning_steps=0` no
hay actualizaciones simuladas y el agente funciona como control muy próximo a
Q-Learning tabular. Por el criterio pedido para Dyna-Q, una truncación se guarda
como terminal; el Q-Learning existente sólo elimina el bootstrap ante
`terminated`.

## Experimentos

El lote definido compara `planning_steps = 0, 5, 10, 20, 50`. Para producir los
resultados definitivos se debe ejecutar:

```bash
poetry run python scripts/train_dyna_q.py \
  --planning-steps 0 5 10 20 50 \
  --episodes 20000 --evaluation-episodes 100 \
  --seeds 42 123 999 2026 777
poetry run python scripts/generate_dyna_q_reports.py
```

Los resultados científicos todavía no fueron ejecutados por Codex; la
validación corta se realizó fuera de `results/` para no mezclarlos.

- Resultado por cantidad de pasos: **[AGREGAR RESULTADO]**
- Tabla comparativa: **[AGREGAR TABLA]**
- Curvas de aprendizaje: **[AGREGAR GRÁFICO]**
- Modelo final y justificación: **[EXPLICAR ELECCIÓN FINAL]**

El script selecciona `dyna_q_best.pkl` dentro de cada lote priorizando, en este
orden, tasa de éxito, recompensa promedio y menor cantidad de pasos. Esa
selección automática debe validarse con las distintas semillas antes de elegir
el modelo del informe.

En la tabla comparativa, la recompensa de entrenamiento resume los últimos 100
episodios de cada experimento, igual que el CSV de Q-Learning. “Mejor recompensa”
es la mejor recompensa promedio de evaluación entre las seeds, no el máximo de
un episodio aislado.

## Material sugerido para el informe

- `comparison_qlearning_dynaq.csv`, aclarando que Q-Learning usa el baseline
  `baseline_40x40_a11`.
- Curvas de recompensa y promedio móvil para observar velocidad y estabilidad.
- Gráfico de recompensa greedy contra `planning_steps`.
- Gráfico de tiempo de entrenamiento contra `planning_steps` para mostrar el
  costo computacional de planificar.
- Captura del comando, `run_id`, cantidad de seeds y archivos generados.

## Texto breve para reutilizar

Dyna-Q fue implementado como una extensión de Q-Learning tabular. Para permitir
una comparación justa, se reutilizó la misma discretización de estados y
acciones seleccionada en los mejores experimentos de Q-Learning. La diferencia
principal fue la incorporación de un modelo interno del ambiente, donde se
almacenan transiciones observadas durante la interacción real. Luego de cada
paso real, el agente ejecuta una cantidad configurable de pasos de
planificación, actualizando la Q-table a partir de experiencias simuladas. Se
definieron experimentos con 0, 5, 10, 20 y 50 pasos de planificación para
analizar su impacto sobre la velocidad de aprendizaje, el rendimiento greedy y
el tiempo de entrenamiento.
