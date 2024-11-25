import numpy as np
import matplotlib.pyplot as plt

# Definición de la función de Rosenbrock y su gradiente
def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def grad_rosenbrock(x, y, a=1, b=100):
    df_dx = -2 * (a - x) - 4 * b * x * (y - x**2)
    df_dy = 2 * b * (y - x**2)
    return np.array([df_dx, df_dy])

# Implementación del algoritmo de gradiente descendente
def gradient_descent(initial_point, learning_rate, tol=1e-6, max_iter=10000, a=1, b=100):
    x, y = initial_point
    trajectory = [(x, y)]
    grad_norms = []
    values = []
    
    for _ in range(max_iter):
        grad = grad_rosenbrock(x, y, a, b)
        grad_norms.append(np.linalg.norm(grad))
        values.append(rosenbrock(x, y, a, b))
        
        if np.linalg.norm(grad) < tol:  # Criterio de convergencia
            break
        x, y = np.array([x, y]) - learning_rate * grad
        trajectory.append((x, y))
    
    return np.array(trajectory), values, grad_norms

# Configuración para análisis de tasas de aprendizaje y condiciones iniciales
learning_rates = [0.001, 0.01, 0.1, 0.2]
initial_points = [(1.5, 1.5), (-1.5, 2.0), (0.0, 0.0)]
tol = 1e-6
max_iter = 10000

# Visualización de trayectorias en contornos de la función
x_range = np.linspace(-2, 2, 400)
y_range = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock(X, Y)

plt.figure(figsize=(12, 8))
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')

for lr in learning_rates:
    for initial_point in initial_points:
        trajectory, _, _ = gradient_descent(initial_point, lr, tol, max_iter)
        plt.plot(trajectory[:, 0], trajectory[:, 1], label=f"LR={lr}, Init={initial_point}")
        plt.scatter(*trajectory[-1], label=f"Final Point LR={lr}, Init={initial_point}")

plt.scatter(1, 1, color="red", label="Global Min (1, 1)", s=100)
plt.title("Trayectorias en Contornos de la Función de Rosenbrock")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="best", fontsize=8)
plt.grid()
plt.show()

# Evolución del valor de la función objetivo
plt.figure(figsize=(12, 8))
for lr in learning_rates:
    for initial_point in initial_points:
        _, values, _ = gradient_descent(initial_point, lr, tol, max_iter)
        plt.plot(values, label=f"LR={lr}, Init={initial_point}")

plt.title("Evolución del Valor de la Función Objetivo")
plt.xlabel("Iteraciones")
plt.ylabel("f(x, y)")
plt.yscale("log")  # Escala logarítmica para visualizar mejor
plt.legend(loc="best", fontsize=8)
plt.grid()
plt.show()

# Magnitud del gradiente a lo largo de las iteraciones
plt.figure(figsize=(12, 8))
for lr in learning_rates:
    for initial_point in initial_points:
        _, _, grad_norms = gradient_descent(initial_point, lr, tol, max_iter)
        plt.plot(grad_norms, label=f"LR={lr}, Init={initial_point}")

plt.title("Magnitud del Gradiente a lo Largo de las Iteraciones")
plt.xlabel("Iteraciones")
plt.ylabel("||∇f(x, y)||")
plt.yscale("log")  # Escala logarítmica para visualizar mejor
plt.legend(loc="best", fontsize=8)
plt.grid()
plt.show()
