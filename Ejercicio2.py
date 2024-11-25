import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from scipy.linalg import svd

# Cargar el dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Agregar columna de unos para el término independiente
X = np.hstack((np.ones((X.shape[0], 1)), X))

# Particionar el dataset en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Estandarizar las características usando solo el conjunto de entrenamiento
mean = X_train[:, 1:].mean(axis=0)
std = X_train[:, 1:].std(axis=0)
std[std == 0] = 1  # Evitar división por cero
X_train[:, 1:] = (X_train[:, 1:] - mean) / std
X_test[:, 1:] = (X_test[:, 1:] - mean) / std

# ------------------------------------------------------------------------------------
# 1. Solución analítica mediante pseudoinversa
# ------------------------------------------------------------------------------------
# w = (XᵗX)⁻¹Xᵗy
w_pinv = np.linalg.pinv(X_train) @ y_train

# Cálculo del ECM en entrenamiento y prueba
n_train = X_train.shape[0]
n_test = X_test.shape[0]
ecm_train_pinv = (1 / n_train) * np.linalg.norm(y_train - X_train @ w_pinv) ** 2
ecm_test_pinv = (1 / n_test) * np.linalg.norm(y_test - X_test @ w_pinv) ** 2

print("Solución mediante Pseudoinversa:")
print(f"ECM Entrenamiento: {ecm_train_pinv:.4f}")
print(f"ECM Prueba: {ecm_test_pinv:.4f}")

# ------------------------------------------------------------------------------------
# 2. Implementación del gradiente descendente
# ------------------------------------------------------------------------------------
# Calcular el valor singular máximo de X_train
_, S, _ = svd(X_train, full_matrices=False)
sigma1 = S[0]

# Definir diferentes tasas de aprendizaje η
etas = [1 / sigma1 ** 2, 0.1 / sigma1 ** 2, 0.01 / sigma1 ** 2]
eta_labels = ['1/σ₁²', '0.1/σ₁²', '0.01/σ₁²']

iterations = 10000
n_features = X_train.shape[1]

# Listas para almacenar los errores y diferencias
train_errors_all = []
test_errors_all = []
w_diffs = []

for eta in etas:
    # Inicializar parámetros
    w = np.zeros(n_features)
    train_errors = []
    test_errors = []

    for _ in range(iterations):
        # Calcular gradiente
        gradient = (2 / n_train) * X_train.T @ (X_train @ w - y_train)
        # Actualizar parámetros
        w -= eta * gradient
        # Calcular y almacenar el error
        ecm_train = (1 / n_train) * np.linalg.norm(y_train - X_train @ w) ** 2
        ecm_test = (1 / n_test) * np.linalg.norm(y_test - X_test @ w) ** 2
        train_errors.append(ecm_train)
        test_errors.append(ecm_test)

    train_errors_all.append(train_errors)
    test_errors_all.append(test_errors)
    # Calcular diferencia con la solución analítica
    w_diff = np.linalg.norm(w - w_pinv)
    w_diffs.append(w_diff)

# ------------------------------------------------------------------------------------
# 3. Comparación y análisis de resultados
# ------------------------------------------------------------------------------------

# Graficar ECM con puntos destacados para convergencia
tolerancia = 1e-6  # Definir tolerancia de convergencia
convergencia_iters = []

plt.figure(figsize=(14, 7))
for idx, eta in enumerate(etas):
    # Graficar errores de entrenamiento y prueba
    plt.plot(train_errors_all[idx], label=f'Entrenamiento η = {eta_labels[idx]}')
    plt.plot(test_errors_all[idx], linestyle='--', label=f'Prueba η = {eta_labels[idx]}')

    # Encontrar iteración de convergencia
    convergencia = next((i for i, e in enumerate(train_errors_all[idx]) if e < tolerancia), iterations)
    convergencia_iters.append(convergencia)
    if convergencia < iterations:
        plt.scatter(convergencia, train_errors_all[idx][convergencia], color='red', label=f'Convergencia η = {eta_labels[idx]}')

# Líneas horizontales para la solución de pseudoinversa
plt.axhline(y=ecm_train_pinv, color='r', linestyle='-', label='Pseudoinversa Entrenamiento')
plt.axhline(y=ecm_test_pinv, color='k', linestyle='-', label='Pseudoinversa Prueba')

plt.xlabel('Iteraciones')
plt.ylabel('ECM')
plt.title('Comparación de ECM: Gradiente Descendente vs Pseudoinversa')
plt.legend()
plt.grid(True)
plt.show()

# Comparar ECM final entre pseudoinversa y gradiente descendente
ecm_final_train = [errors[-1] for errors in train_errors_all]
ecm_final_test = [errors[-1] for errors in test_errors_all]

print("\nComparación de ECM final:")
print(f"Pseudoinversa - ECM Entrenamiento: {ecm_train_pinv:.6f}, ECM Prueba: {ecm_test_pinv:.6f}")
for idx, eta in enumerate(etas):
    print(f"η = {eta_labels[idx]}")
    print(f"ECM Entrenamiento final: {ecm_final_train[idx]:.6f}")
    print(f"ECM Prueba final: {ecm_final_test[idx]:.6f}")

# ------------------------------------------------------------------------------------
# Comparar soluciones de pseudoinversa y gradiente descendente al final de las iteraciones
# ------------------------------------------------------------------------------------

plt.figure(figsize=(12, 6))

# Graficar diferencias en ECM final
ecm_final_train = [errors[-1] for errors in train_errors_all]
ecm_final_test = [errors[-1] for errors in test_errors_all]

# Gráfico de barras para ECM final en entrenamiento y prueba
x = np.arange(len(etas))
width = 0.35  # Ancho de las barras

plt.bar(x - width/2, ecm_final_train, width, label='Gradiente Descendente - Entrenamiento', color='blue')
plt.bar(x + width/2, ecm_final_test, width, label='Gradiente Descendente - Prueba', color='red')

# Línea horizontal para ECM de pseudoinversa
plt.axhline(y=ecm_train_pinv, color='red', linestyle='-', label='Pseudoinversa - Entrenamiento')
plt.axhline(y=ecm_test_pinv, color='black', linestyle='--', label='Pseudoinversa - Prueba')

# Etiquetas y leyenda
plt.xticks(x, eta_labels)
plt.xlabel('Tasa de Aprendizaje (η)')
plt.ylabel('ECM Final')
plt.title('Comparación de ECM Final: Pseudoinversa vs Gradiente Descendente')
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------------------------------------------------------------
# Comparación de diferencias ||w_grad - w_pinv||
# ------------------------------------------------------------------------------------

plt.figure(figsize=(12, 6))

# Graficar la diferencia ||w_grad - w_pinv||
plt.bar(eta_labels, w_diffs, color='green', alpha=0.7)

# Etiquetas y leyenda
plt.xlabel('Tasa de Aprendizaje (η)')
plt.ylabel('Diferencia ||w_grad - w_pinv||')
plt.title('Diferencia entre Coeficientes: Gradiente Descendente vs Pseudoinversa')
plt.grid(True)
plt.show()

