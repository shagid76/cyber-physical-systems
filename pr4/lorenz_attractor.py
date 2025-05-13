import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

def lorenz_deriv(t, xyz):
    x, y, z = xyz
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

xyz0 = [1.0, 1.0, 1.0]
t_span = (0, 100)
t_eval = np.linspace(0, 100, 10000)

sol = solve_ivp(lorenz_deriv, t_span, xyz0, t_eval=t_eval)

def plot_lorenz_attractor():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(sol.y[0], sol.y[1], sol.y[2], 'b-', alpha=0.6)
    ax.set_title('Lorenz Attractor')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__ == "__main__":
    plot_lorenz_attractor()