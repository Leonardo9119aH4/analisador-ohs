import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# PARTE 1: CALCULAR FATOR DE QUALIDADE

df = pd.read_csv("dados_pendulo.csv")
t = df["tempo_s"].values
x = df["pos_x"].values
y = df["pos_y"].values
picos, _ = find_peaks(x, distance=30, prominence=30) # Detecta valores máximos da função
vales, _ = find_peaks(-x, distance=30, prominence=25) # Detecta valores mínimos da função
t_amp = [] # Tempos dos picos e vales
amplitudes = [] 

for p in picos[1:-1]: # Preenche os arrays de vales e picos da função com o valor de x e seu respectivo tempo
    antes = vales[vales < p]
    depois = vales[vales > p]
    if len(antes) > 0 and len(depois) > 0:
        vale1 = antes[-1]
        vale2 = depois[0]
        A = (x[p] - (x[vale1] + x[vale2]) / 2) / 2
        if A > 0:
            t_amp.append(t[p])
            amplitudes.append(A)
t_amp = np.array(t_amp)
amplitudes = np.array(amplitudes)

def envelope(t, A0, gamma):
    return A0 * np.exp(-gamma * t)
popt, _ = curve_fit(
    envelope, t_amp, amplitudes,
    p0=[amplitudes.max(), 0.003] # Chute inicial para o curve_fit
)
A0, gamma = popt

periodo = np.mean(np.diff(t[picos])) # Média entre os tempos entre cada pico
omega = 2 * np.pi / periodo
Q = omega / (2 * gamma)

print(f"A0 = {A0:.2f} pixels")
print(f"gamma = {gamma:.6f} s^-1")
print(f"Período = {periodo:.4f} s")
print(f"omega = {omega:.4f} rad/s")
print(f"Q = {Q:.2f}")

# PARTE 2: GRÁFICOS

vx = np.gradient(x, t)
vy = np.gradient(y, t)
# Posição x
plt.figure(figsize=(10, 5))
plt.plot(t, x)
plt.title("Posição x em função do tempo")
plt.xlabel("Tempo (s)")
plt.ylabel("x (pixels)")
plt.grid(True)
plt.show()
# Posição y
plt.figure(figsize=(10, 5))
plt.plot(t, y)
plt.title("Posição y em função do tempo")
plt.xlabel("Tempo (s)")
plt.ylabel("y (pixels)")
plt.grid(True)
plt.show()
# Velocidade vx
plt.figure(figsize=(10, 5))
plt.plot(t, vx)
plt.title("Velocidade vx em função do tempo")
plt.xlabel("Tempo (s)")
plt.ylabel("vx (pixels/s)")
plt.grid(True)
plt.show()
# Velocidade vy
plt.figure(figsize=(10, 5))
plt.plot(t, vy)
plt.title("Velocidade vy em função do tempo")
plt.xlabel("Tempo (s)")
plt.ylabel("vy (pixels/s)")
plt.grid(True)
plt.show()

