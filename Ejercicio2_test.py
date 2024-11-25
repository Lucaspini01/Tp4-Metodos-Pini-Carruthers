import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from scipy.linalg import svd

# Cargar el dataset
housing = fetch_california_housing()
X = housing.data
y = housing.target

# La regresión lineal busca minimizar la funcion objetivo del error cuadrático medio (ECM) de las predicciones ˆy = Xw, definido como:
def ECM(w,n):
    return 1/n * np.linalg.norm(y - X @ w) **2

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

#---PSEUDOINVERSA---------------------------

# Implementar la solución analítica de mínimos cuadrados usando pseudoinversa
w_analitico = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Calcular el error cuadrático medio en entrenamiento y prueba
n_train = X_train.shape[0]
n_test = X_test.shape[0]
ecm_train_analitico = (1 / n_train) * np.linalg.norm(y_train - X_train @ w_analitico) ** 2
ecm_test_analitico = (1 / n_test) * np.linalg.norm(y_test - X_test @ w_analitico) ** 2

print("Solución analítica de mínimos cuadrados con Pseudoinversa:")
print(f"ECM Entrenamiento: {ecm_train_analitico:.4f}")
print(f"ECM Prueba: {ecm_test_analitico:.4f}")

#---GRADIENTE DECRECIENTE-----------------------

# Implementar el algoritmo de gradiente decreciente para minimizar el error cuadratico medio

# Calcular el valor singular máximo de X_train
_, S, _ = svd(X_train, full_matrices=False)
sigma1 = S[0]

# Definir tasa de aprendizaje
eta = 1 / sigma1**2

# Inicializar parámetros
w_grad = np.zeros(X_train.shape[1])
iterations = 1000
n_train = X_train.shape[0]

# Almacenar el error en cada iteración
train_errors_grad = []
test_errors_grad = []

for i in range(iterations):
    # Calcular gradiente
    gradient = (2 / n_train) * X_train.T @ (X_train @ w_grad - y_train)
    # Actualizar parámetros
    w_grad -= eta * gradient
    # Calcular y almacenar el error
    ecm_train_grad = (1 / n_train) * np.linalg.norm(y_train - X_train @ w_grad) ** 2
    ecm_test_grad = (1 / n_test) * np.linalg.norm(y_test - X_test @ w_grad) ** 2
    train_errors_grad.append(ecm_train_grad)
    test_errors_grad.append(ecm_test_grad)

# # Graficar el ECM a lo largo de las iteraciones
# plt.figure(figsize=(12, 6))
# plt.plot(range(iterations), train_errors_grad, label='Entrenamiento')
# plt.plot(range(iterations), test_errors_grad, linestyle='--', label='Prueba')
# plt.xlabel('Iteraciones')
# plt.ylabel('ECM')
# plt.title('Error Cuadrático Medio vs Iteraciones para Gradiente Descendente')
# plt.legend()
# plt.grid(True)
# plt.show()

# Imprimir resultados finales
print("Solución mediante Gradiente Descendente:")
print(f"ECM Entrenamiento final: {train_errors_grad[-1]:.4f}")
print(f"ECM Prueba final: {test_errors_grad[-1]:.4f}")

#---ANALSIS------------------------------------

#Comparar la solucion obtenida por la pseudoinversa con la solucion iteratica del gradiente descendente para distitos valores de eta

etas = [1 / sigma1**2, 0.1 / sigma1**2, 0.01 / sigma1**2, 0.001 / sigma1**2, 0.0001 / sigma1**2]
eta_labels = ['1/σ₁²', '0.1/σ₁²', '0.01/σ₁²', '0.001/σ₁²', '0.0001/σ₁²']

iterations = 1000
n_features = X_train.shape[1]

w_analiticos = []
w_gradiente_decreciente = []

for eta in etas:
    w_analitico = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
    w_analiticos.append(w_analitico)

    w_gradiente_decreciente = np.zeros(n_features)
    for _ in range(iterations):
        # Calcular gradiente
        gradient = (2 / n_train) * X_train.T @ (X_train @ w_gradiente_decreciente - y_train)
        # Actualizar parámetros
        w_gradiente_decreciente -= eta * gradient
    
    w_gradiente_decreciente.append(w_gradiente_decreciente)

plt.figure(figsize=(14, 7))
for idx, eta in enumerate(etas):
    plt.plot(w_analiticos[idx], label=f'Analítica η = {eta_labels[idx]}')
    plt.plot(w_gradiente_decreciente[idx], linestyle='--', label=f'Gradiente η = {eta_labels[idx]}')

plt.xlabel('Parámetros')
plt.ylabel('Valor')
plt.title('Comparación de Parámetros: Pseudoinversa vs Gradiente Descendente')
plt.legend()
plt.grid(True)
plt.show()


