import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import animation

EMPTY = 0
TREE = 1
BURNING = 2

colors = {
    EMPTY: (0.2, 0.2, 0.2),
    TREE: (0.0, 0.6, 0.0),
    BURNING: (1.0, 0.0, 0.0)
}

def create_forest(size, fire_spots):
    forest = np.ones((size, size), dtype=int)
    burning_time = np.zeros_like(forest)
    for x, y in fire_spots:
        forest[x, y] = BURNING
    return forest, burning_time

def update_forest(forest, burning_time, P_burn, T_burn):
    new_forest = forest.copy()
    new_burning_time = burning_time.copy()
    size = forest.shape[0]

    for x in range(size):
        for y in range(size):
            if forest[x, y] == TREE:
                neighbors = forest[max(0, x - 1):x + 2, max(0, y - 1):y + 2]
                if np.any(neighbors == BURNING):
                    if np.random.rand() < P_burn:
                        new_forest[x, y] = BURNING
                        new_burning_time[x, y] = 1
            elif forest[x, y] == BURNING:
                new_burning_time[x, y] += 1
                if new_burning_time[x, y] > T_burn:
                    new_forest[x, y] = EMPTY

    return new_forest, new_burning_time

def visualize_simulation(size=50, steps=100, P_burn=0.3, T_burn=3):
    fire_spots = [(size // 2, size // 2)]
    forest, burning_time = create_forest(size, fire_spots)

    fig = plt.figure(figsize=(6, 6))
    img = plt.imshow(np.zeros((size, size, 3)), interpolation='nearest')

    def animate(i):
        nonlocal forest, burning_time
        forest, burning_time = update_forest(forest, burning_time, P_burn, T_burn)
        rgb_forest = np.zeros((size, size, 3))
        for state in [EMPTY, TREE, BURNING]:
            rgb_forest[forest == state] = colors[state]
        img.set_array(rgb_forest)
        return [img]

    anim = animation.FuncAnimation(fig, animate, frames=steps, interval=200, blit=True)
    plt.title(f"Forest Fire Model\nP_burn={P_burn}, T_burn={T_burn}")
    plt.axis('off')
    plt.show()

visualize_simulation(size=50, steps=100, P_burn=0.8, T_burn=4)