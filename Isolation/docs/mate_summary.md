# MATE — resumen para el informe

## Implementación

Cada acción se obtiene con `Board.get_possible_actions(player)`. Esta función
genera el producto entre movimientos válidos y casillas eliminables **después de
mover**; por eso no se puede eliminar la nueva posición ni una casilla ocupada.
Para explorar el árbol, cada algoritmo clona el tablero y aplica la acción sobre
la copia. El tablero real nunca se modifica durante la búsqueda.

- **Minimax:** alterna nodos MAX (nuestro agente) y MIN (rival), hasta alcanzar
  un estado terminal o el límite de profundidad.
- **Alpha-Beta:** calcula el mismo valor que Minimax, pero mantiene cotas `alpha`
  y `beta`. Cuando una rama ya no puede cambiar la decisión, omite sus sucesores.
  Se registran tanto nodos visitados como ramas podadas.
- **Expectimax:** usa MAX para nuestro agente y trata el turno rival como un nodo
  de azar uniforme. El valor del rival es el promedio de todos sus sucesores.
  No se usa poda Alpha-Beta porque no es válida, en general, para promedios.

Los terminales valen `+1` si gana el agente y `-1` si pierde. Las heurísticas no
terminales se normalizan al intervalo `(-1, 1)`, de modo que nunca sean mejores
que una victoria demostrada.

## Funciones de evaluación

`mate_evaluations.py` incluye seis alternativas seleccionables:

1. `mobility`: diferencia entre movimientos propios y rivales.
2. `aggressive`: penaliza la movilidad rival.
3. `defensive`: premia la movilidad propia.
4. `balanced`: da peso doble a la movilidad propia.
5. `territory`: compara casillas libres alcanzables.
6. `weighted`: combina movilidad, casillas bloqueadas alrededor de ambos
   jugadores y distancia entre ellos.

Se cuentan direcciones de movimiento (máximo ocho), no todas las combinaciones
movimiento-eliminación.

## Experimentos y métricas

`experiments_mate.py` alterna qué agente comienza para reducir el sesgo del
primer jugador y usa semillas reproducibles. Compara los tres algoritmos contra
Random, Alpha-Beta contra Minimax y Expectimax, distintas profundidades, y las
funciones de evaluación. El modo `full` agrega un todos-contra-todos de
heurísticas.

Por configuración se guardan victorias, derrotas, tasa de victoria, longitud
media, tiempo medio de decisión, nodos medios visitados y ramas medias podadas.
Los resultados quedan en `results/mate_experiments.csv` y
`results/mate_experiments.json`.

La ejecución de validación incluida (4x4, dos partidas por configuración,
semilla 2026) muestra el efecto esperado a profundidad 2 en el enfrentamiento
Alpha-Beta–Minimax: aproximadamente 292 nodos visitados por decisión para
Alpha-Beta frente a 1674 para Minimax. Alpha-Beta omitió en promedio 1278 ramas
directas por decisión. Los números cambian con la semilla y la cantidad de
partidas, pero ambos algoritmos conservan la misma regla de decisión Minimax.

No se generan gráficos automáticamente porque el proyecto no declara una
dependencia de visualización. El CSV queda listo para graficar en una planilla o
en el notebook.
