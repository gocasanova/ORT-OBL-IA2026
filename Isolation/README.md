
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
