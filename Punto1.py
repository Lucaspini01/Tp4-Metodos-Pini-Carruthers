import numpy as np
import matplotlib.pyplot as plt

#Condiciones iniciales
learning_rates = [0.001 , 0.002, 0.005]

# Definición de la función de Rosenbrock y su gradiente
def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def grad_rosenbrock(x, y, a=1, b=100):
    df_dx = -2 * (a - x) - 4 * b * x * (y - x**2)
    df_dy = 2 * b * (y - x**2)
    return np.array([df_dx, df_dy])

# Implementación del algoritmo de gradiente descendente
def gradient_descent(initial_point, learning_rate, tol=1e-6, max_iter=1000, a=1, b=100):
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

# Parámetros del algoritmo
initial_point = np.array([-1.5, 1.5])
learning_rate = learning_rates[0]
tol = 1e-15
max_iter = 80000

# Ejecución del algoritmo
trajectory, values, grad_norms = gradient_descent(initial_point, learning_rate, tol, max_iter)

# Paramtros del segundo algoritmo
initial_point = np.array([-1.5, 1.5])
learning_rate = learning_rates[1]
tol = 1e-15
max_iter = 80000

# Ejecución del algoritmo
trajectory2, values2, grad_norms2 = gradient_descent(initial_point, learning_rate, tol, max_iter)


# Gráfica de la función de Rosenbrock
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

#Copiar los parametro 

plt.figure(figsize=(10, 6))
plt.contour(X, Y, Z, levels=50, cmap='viridis')
plt.plot(*zip(*trajectory), color='red', marker='o', markersize=5)
plt.plot(*zip(*trajectory2), color='blue', marker='o', markersize=5)
plt.title('Función de Rosenbrock y trayectoria del gradiente descendente')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Valor de la función de Rosenbrock')
plt.show()

# Ensure trajectory and values have the same length
min_length = min(len(trajectory), len(values))
trajectory = trajectory[:min_length]
values = values[:min_length]

# Ensure trajectory and values have the same length
min_length = min(len(trajectory2), len(values2))
trajectory2 = trajectory2[:min_length]
values2 = values2[:min_length]


# Debugging: Check shapes of trajectory and values
print(f"Shape of trajectory: {len(trajectory)}")
print(f"Shape of values: {len(values)}")

# Gráficar en 3D  
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_title('Función de Rosenbrock')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()


#Graficar la evolucion del descenso por gradiente
plt.figure(figsize=(10, 6))
plt.plot(values, label='Trajectory 1', color='red')
plt.plot(values2, label='Trajectory 2', color='blue')
plt.yscale('log')
plt.title('Evolución del descenso por gradiente')
plt.xlabel('Iteración')
plt.ylabel('Valor de la función de Rosenbrock')
plt.legend()
plt.show()

