"""
Guía de Práctica de Laboratorio 06
Desarrollo, implementación y versionamiento de redes recurrentes (RNN y LSTM)
Autor: Rodrigo Fabricio Auccapiña Bombilla
Escuela Profesional de Ingeniería de Sistemas - Universidad Andina del Cusco
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

SEED = 42
EPOCHS = 50
LR = 0.01
CARPETA_RESULTADOS = "resultados"
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Generación de datos sintéticos (serie temporal de una onda senoidal)
# ---------------------------------------------------------------------------
def generar_datos(n_puntos=500, seq_len=10):
    t = np.linspace(0, 50, n_puntos)
    serie = np.sin(t)

    X, y = [], []
    for i in range(len(serie) - seq_len):
        X.append(serie[i:i + seq_len])
        y.append(serie[i + seq_len])

    # np.array primero: convertir una lista de arrays directo a tensor es lento
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)  # [N, seq_len, 1]
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)  # [N, 1]
    return X, y


# ---------------------------------------------------------------------------
# 2. Arquitecturas
# ---------------------------------------------------------------------------
class SimpleRNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super(SimpleRNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)          # out: [batch, seq_len, hidden]
        out_final = out[:, -1, :]     # último paso temporal (Many-to-One)
        return self.fc(out_final)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)  # LSTM retorna estado oculto y estado de celda
        out_final = out[:, -1, :]
        return self.fc(out_final)


# ---------------------------------------------------------------------------
# 3. Utilidades
# ---------------------------------------------------------------------------
def contar_parametros(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def entrenar(model, X, y, epochs=EPOCHS, lr=LR, nombre="Modelo"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    historial = []

    for epoch in range(epochs):
        model.train()
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        historial.append(loss.item())
        if (epoch + 1) % 10 == 0:
            print(f"  [{nombre}] Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

    return historial


def predecir(model, X):
    model.eval()
    with torch.no_grad():
        return model(X)


def nuevo_modelo(clase, hidden_size=16):
    torch.manual_seed(SEED)  # misma inicialización para comparar en igualdad de condiciones
    return clase(input_size=1, hidden_size=hidden_size, output_size=1)


# ---------------------------------------------------------------------------
# 4. Actividades prácticas
# ---------------------------------------------------------------------------
def actividad_1_dimensiones():
    print("\n=== ACTIVIDAD 1: Inspección de dimensiones (tensores 3D) ===")
    X, y = generar_datos(n_puntos=600, seq_len=10)
    print(f"X.shape = {tuple(X.shape)}  -> [Batch Size, Sequence Length, Features]")
    print(f"y.shape = {tuple(y.shape)}  -> [Batch Size, Output]")
    print(f"  Batch Size      = {X.shape[0]} ventanas (600 puntos - 10 de longitud)")
    print(f"  Sequence Length = {X.shape[1]} pasos temporales por ventana")
    print(f"  Features        = {X.shape[2]} variable por paso (el valor de sin(t))")


def actividad_2_parametros():
    print("\n=== ACTIVIDAD 2: Parámetros entrenables según hidden_size ===")
    for clase in (SimpleRNNModel, LSTMModel):
        for h in (16, 64):
            m = nuevo_modelo(clase, hidden_size=h)
            print(f"  {clase.__name__:15s} hidden_size={h:3d} -> {contar_parametros(m):6d} parámetros")


def actividad_3_ventanas():
    print("\n=== ACTIVIDAD 3: Efecto de SEQ_LEN en la RNN simple ===")
    historiales = {}
    for seq_len in (10, 50):
        X, y = generar_datos(n_puntos=600, seq_len=seq_len)
        print(f" SEQ_LEN = {seq_len}  X.shape = {tuple(X.shape)}")
        model = nuevo_modelo(SimpleRNNModel)
        historiales[seq_len] = entrenar(model, X, y, nombre=f"RNN seq={seq_len}")

    plt.figure(figsize=(8, 5))
    for seq_len, h in historiales.items():
        plt.plot(h, label=f"RNN simple, SEQ_LEN={seq_len}")
    plt.yscale("log")
    plt.xlabel("Época")
    plt.ylabel("Loss (MSE, escala log)")
    plt.title("Actividad 3: RNN simple con SEQ_LEN 10 vs 50")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_RESULTADOS, "actividad3_seq_len.png"), dpi=120)
    plt.close()
    return historiales


# ---------------------------------------------------------------------------
# 5. Reto avanzado: RNN simple vs LSTM con SEQ_LEN = 50
# ---------------------------------------------------------------------------
def reto_rnn_vs_lstm(seq_len=50):
    print(f"\n=== RETO: RNN simple vs LSTM (SEQ_LEN = {seq_len}) ===")
    X, y = generar_datos(n_puntos=600, seq_len=seq_len)

    rnn = nuevo_modelo(SimpleRNNModel)
    lstm = nuevo_modelo(LSTMModel)

    hist_rnn = entrenar(rnn, X, y, nombre="RNN ")
    hist_lstm = entrenar(lstm, X, y, nombre="LSTM")

    pred_rnn = predecir(rnn, X)
    pred_lstm = predecir(lstm, X)
    mse = nn.MSELoss()
    mse_rnn = mse(pred_rnn, y).item()
    mse_lstm = mse(pred_lstm, y).item()

    # Curvas de convergencia
    plt.figure(figsize=(8, 5))
    plt.plot(hist_rnn, label="RNN simple")
    plt.plot(hist_lstm, label="LSTM")
    plt.yscale("log")
    plt.xlabel("Época")
    plt.ylabel("Loss (MSE, escala log)")
    plt.title(f"Convergencia de la pérdida (SEQ_LEN={seq_len})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_RESULTADOS, "reto_loss_rnn_vs_lstm.png"), dpi=120)
    plt.close()

    # Predicciones vs serie real
    plt.figure(figsize=(10, 5))
    plt.plot(y.squeeze().numpy(), label="Serie real", linewidth=2, color="black")
    plt.plot(pred_rnn.squeeze().numpy(), label="Predicción RNN", linestyle="--")
    plt.plot(pred_lstm.squeeze().numpy(), label="Predicción LSTM", linestyle=":")
    plt.xlabel("Índice de la ventana")
    plt.ylabel("sin(t)")
    plt.title(f"Predicción del siguiente valor (SEQ_LEN={seq_len})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_RESULTADOS, "reto_predicciones.png"), dpi=120)
    plt.close()

    print("\n  Resumen final:")
    print(f"  RNN simple -> Loss final: {hist_rnn[-1]:.6f} | MSE predicción: {mse_rnn:.6f}")
    print(f"  LSTM       -> Loss final: {hist_lstm[-1]:.6f} | MSE predicción: {mse_lstm:.6f}")
    mejor = "LSTM" if mse_lstm < mse_rnn else "RNN simple"
    print(f"  Mejor ajuste a la serie: {mejor}")


if __name__ == "__main__":
    actividad_1_dimensiones()
    actividad_2_parametros()
    actividad_3_ventanas()
    reto_rnn_vs_lstm(seq_len=50)
    print(f"\nGráficas guardadas en la carpeta '{CARPETA_RESULTADOS}/'")
    print("¡Entrenamiento de la RNN Simple y LSTM finalizado con éxito!")
