from code import dag as t
import matplotlib.pyplot as plt
import numpy as np
type num = t.num


def generate_pascals_triangle(n: int) -> list[int]:
    """Generate level of pascal triangle using the property C(n,i+1) = C(n,i) * (n-i) / (i+1)"""
    row: list[int] = [1]
    for i in range(n):
        row.append(row[i] * (n - i) // (i + 1))
    return row


def normalize_level(level: list | t.Level):
    """Normalize row n of Pascal's triangle to match standard normal curve"""
    # Create x positions and normalized y values
    # std_dev = math.sqrt(n / 4)
    # x_values = [(k - n / 2) / std_dev for k in range(n + 1)]
    # x_values = [(k - n/2) for k in range(n + 1)]
    x_values: list[int] = [k for k in range(len(level))]

    # Apply vertical normalization
    max_elem: num = max(level)
    y_values = [coefficient / max_elem for coefficient in level]

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
    n: int = len(level)
    x_level, y_level = normalize_level(level)

    # Generate standard normal curve for comparison
    l_idx: int = find_visible_index(y_level)
    r_idx: int = find_visible_index(y_level, reverse=True)

    x_normal = np.linspace(l_idx, r_idx, 100)
    mu: num = y_level.index(1)  # n / 2 for pascals triangle
    v: num = n / 2
    y_normal = np.exp(-((x_normal - mu) ** 2) / v)

    plt.figure(figsize=(10, 6))
    plt.plot(x_level[l_idx:r_idx], y_level[l_idx:r_idx], 'ro', label=f'Dag Level {n} (normalized)')
    plt.plot(x_normal, y_normal, 'b-', label='Gaussian Function')
    plt.title(f'DAG\'s Level {n} vs. Bell Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    import random as r

    basin = [r.randint(1, 15) for _ in range(r.randint(4, 8))]
    # basin = [12, 4, 11, 4, 12, 14, 11, 1]
    dag = t.DAG(basin=t.Level(basin),
                depth=100,  # depth of recursion
                node_arity=2,  # max number of nodes to be used as arguments for the operator (n-ary graph as a result)
                func=t.Funcs.sum,  # to be used in the operator to create the next level of nodes based.
                step_caller=t.print_coefficients,
                save_levels=True)

    # Plot
    plot_comparison(dag.getLevel(5))
    print(basin)

    # plot_comparison(generate_pascals_triangle(10))
