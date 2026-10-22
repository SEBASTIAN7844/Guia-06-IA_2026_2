# Práctica 06: Redes Recurrentes (RNN y LSTM) en PyTorch

**Curso:** Inteligencia Artificial / Aprendizaje Profundo
**Alumno:** Rodrigo Fabricio Auccapiña Bombilla
**Escuela Profesional de Ingeniería de Sistemas, Universidad Andina del Cusco**

## Objetivo

Implementar y comparar una RNN simple y una LSTM para predecir el siguiente valor de una serie temporal senoidal, analizando el efecto del tamaño del estado oculto y de la longitud de la secuencia, y versionar el proyecto con Git y GitHub.

## Estructura del repositorio

```
practica-rnn-pytorch/
├── practica_rnn.py      # Modelos, actividades y reto LSTM
├── README.md            # Documentación y análisis
├── .gitignore
└── resultados/          # Gráficas generadas al ejecutar el script
    ├── actividad3_seq_len.png
    ├── reto_loss_rnn_vs_lstm.png
    └── reto_predicciones.png
```

## Ejecución

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install torch matplotlib numpy
python practica_rnn.py
```

Todos los modelos se inicializan con la misma semilla (`SEED = 42`), se entrenan 50 épocas con Adam (lr = 0.01) y pérdida MSE, de modo que la comparación se hace en condiciones idénticas.

## Arquitectura implementada

Ambos modelos siguen el paradigma **Many-to-One**: reciben una ventana de `SEQ_LEN` valores consecutivos de `sin(t)` y predicen el valor siguiente.

| Capa | RNN simple | LSTM |
|---|---|---|
| Recurrente | `nn.RNN(1 → 16, batch_first=True)` | `nn.LSTM(1 → 16, batch_first=True)` |
| Selección | `out[:, -1, :]` (último paso temporal) | `out[:, -1, :]` |
| Salida | `nn.Linear(16 → 1)` | `nn.Linear(16 → 1)` |

La RNN simple actualiza su estado con `h_t = tanh(W_ih·x_t + W_hh·h_{t-1} + b)`, reutilizando los mismos pesos en cada paso (*weight sharing*). La LSTM añade un estado de celda `c_t` y tres compuertas (olvido, entrada y salida) que deciden qué información se conserva, se agrega o se expone en cada paso.

## Actividad 1: Dimensiones de los tensores

Con `n_puntos = 600` y `SEQ_LEN = 10`:

```
X.shape = (590, 10, 1)
y.shape = (590, 1)
```

| Eje | Valor | Significado |
|---|---|---|
| Batch Size | 590 | Número de ventanas deslizantes que se pueden extraer: 600 − 10 |
| Sequence Length | 10 | Pasos temporales que la red recorre antes de predecir |
| Features | 1 | Variables por paso; aquí solo el valor de `sin(t)` |

`y` tiene forma `[590, 1]` porque cada ventana tiene un único objetivo: el valor que viene justo después. Con `SEQ_LEN = 50` la forma pasa a `(550, 50, 1)`: hay menos ventanas, pero cada una es cinco veces más larga.

## Actividad 2: Parámetros entrenables

| Modelo | hidden_size = 16 | hidden_size = 64 | Factor |
|---|---|---|---|
| RNN simple | 321 | 4 353 | ×13.6 |
| LSTM | 1 233 | 17 217 | ×14.0 |

Para la RNN, la capa recurrente tiene `H·I + H·H + 2H` parámetros (pesos de entrada, pesos recurrentes y dos vectores de sesgo) y la capa lineal `H + 1`:

- H = 16: 16 + 256 + 32 = 304, más 17 de la capa lineal, da **321**.
- H = 64: 64 + 4 096 + 128 = 4 288, más 65, da **4 353**.

El crecimiento no es lineal sino aproximadamente cuadrático, porque domina la matriz recurrente `W_hh` de tamaño `H × H`: cuadruplicar `hidden_size` multiplica los parámetros por casi 14. La LSTM tiene exactamente 4 veces los parámetros recurrentes de la RNN (una matriz por cada compuerta más la de la celda candidata), por eso con H = 16 ya pesa casi lo que una RNN de H = 32.

Mayor capacidad no implica mejor resultado: para una senoide de una sola variable, 16 unidades suelen bastar, y 64 aumentan el costo de cómputo y el riesgo de sobreajuste.

## Actividad 3: Longitud de la secuencia (10 vs 50)

Resultados obtenidos (RNN simple):

| SEQ_LEN | Loss época 10 | Loss época 30 | Loss época 50 |
|---|---|---|---|
| 10 | _completar_ | _completar_ | _completar_ |
| 50 | _completar_ | _completar_ | _completar_ |

![Actividad 3](resultados/actividad3_seq_len.png)

**Análisis.** _(ajustar a lo observado)_ Al pasar de 10 a 50 pasos, el gradiente debe retropropagarse a través de 50 multiplicaciones sucesivas por `W_hh` y por la derivada de `tanh` (BPTT). Como esa derivada es menor o igual a 1, el producto tiende a cero y los primeros pasos de la ventana casi no influyen en la actualización de los pesos: es la "amnesia matemática" de los gradientes desvanecientes. En la práctica esto se observa como una pérdida que baja más lento, con más oscilaciones o que se estanca en un valor mayor que con SEQ_LEN = 10.

## Reto: RNN simple vs LSTM con SEQ_LEN = 50

| Modelo | Parámetros | Loss final (época 50) | MSE de predicción |
|---|---|---|---|
| RNN simple | 321 | _completar_ | _completar_ |
| LSTM | 1 233 | _completar_ | _completar_ |

![Curvas de pérdida](resultados/reto_loss_rnn_vs_lstm.png)

![Predicciones](resultados/reto_predicciones.png)

**Análisis.** _(ajustar a lo observado)_ La LSTM evita el desvanecimiento porque su estado de celda se actualiza de forma aditiva (`c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`): cuando la compuerta de olvido está cerca de 1, el gradiente fluye hacia atrás sin multiplicarse repetidamente por factores pequeños. Por eso se espera que la LSTM alcance una pérdida menor y una curva de predicción que siga mejor a la serie real.

Hay dos matices que conviene reportar con honestidad. Primero, la LSTM tiene casi 4 veces más parámetros, así que parte de la ventaja puede venir de su mayor capacidad y no solo de las compuertas. Segundo, con solo 50 épocas la LSTM puede arrancar más lento en las primeras épocas (tiene más pesos que ajustar) y superar a la RNN recién hacia el final; y como una senoide es una señal muy regular, la diferencia puede ser moderada. Con series que dependan de eventos lejanos la brecha sería mayor.

## Conclusiones

1. Los datos de una RNN en PyTorch con `batch_first=True` tienen la forma `[batch, seq_len, features]`, y elegir mal estas dimensiones es el error más común al implementar estas redes.
2. El número de parámetros crece de forma aproximadamente cuadrática con `hidden_size`, y la LSTM cuadruplica los parámetros recurrentes de la RNN.
3. Aumentar la longitud de la secuencia expone el problema de gradientes desvanecientes de la RNN simple.
4. La LSTM, gracias a sus compuertas y a su estado de celda, mantiene mejor la información a largo plazo y se ajusta mejor a la serie con ventanas largas.
