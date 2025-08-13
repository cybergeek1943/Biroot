from src import dag as t
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
type num = int | float


def normalize_level(level: list | t.Level) -> tuple[list[int], list[float]]:
    """Normalize row n of Pascal's triangle to match standard normal curve"""
    # Create x positions and normalized y values
    # std_dev = math.sqrt(n / 4)
    # x_values = [(k - n / 2) / std_dev for k in range(n + 1)]
    # x_values = [(k - n/2) for k in range(n + 1)]
    x_values: list[int] = [k for k in range(len(level))]

    # Apply vertical normalization
    max_elem: num = max(level)
    y_values: list[float] = [coefficient / max_elem for coefficient in level]

    return x_values, y_values


def find_visible_index(level: list | t.Level, reverse: bool = False, threshold: int = 0.001) -> int:
    idx: int = len(level) if reverse else 0
    delta: int = -1 if reverse else 1
    for v in reversed(level) if reverse else level:
        idx += delta
        if v > threshold:
            return idx
    return len(level) if reverse else 0


# Plot both the normalized level and a normal gaussian curve
def plot_comparison(level: list[num] | t.Level) -> None:
    m: int = len(level)
    x_level, y_level = normalize_level(level)

    # Generate standard normal curve for comparison
    l_idx: int = find_visible_index(y_level)
    r_idx: int = find_visible_index(y_level, reverse=True)
    # l_idx = 0
    # r_idx = m

    x_normal = np.linspace(l_idx, r_idx, 100)
    mu: num = y_level.index(1)  # m/2 for pascals triangle
    v: num = m/4
    y_normal = np.exp(-((x_normal - mu) ** 2) / (2*v))

    plt.figure(figsize=(10, 6))
    plt.plot(x_level[l_idx:r_idx], y_level[l_idx:r_idx], 'ro', label=f"Coeffs from DAG level 500")
    plt.plot(x_normal, y_normal, 'b-', label='Gaussian Function')
    # plt.title(f'Normalized Binomial Coefficients vs Gaussian Function')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    import random as r

    # basin = [r.randint(1, 15) for _ in range(r.randint(4, 8))]
    # # basin = [12, 4, 11, 4, 12, 14, 11, 1]
    # dag = t.DAG(basin=t.Level(basin),
    #             depth=100,  # depth of recursion
    #             node_arity=2,  # max number of nodes to be used as arguments for the operator (n-ary graph as a result)
    #             func=t.Funcs.sum,  # to be used in the operator to create the next level of nodes based.,
    #             save_levels=True)

    # ============ Using DAGs ============
    from dag import DAG
    dag = DAG().as_graph(basin=[16, 6, 5, 12, 15, 11, 11, 8, 13, 1, 14, 7], depth=500, node_arity=2)
    l = dag.get_level(500)
    plot_comparison(l)

    # ============ Any List Plot ============
    # l = []
    # plot_comparison(t.Level(l))
