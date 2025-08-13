import numpy as np
from src.biroot import Biroot, GaussianFunctions, DAG
from typing import Callable
from tqdm import tqdm


def load_dataset(filename: str, n: int | None = None) -> dict[int, np.ndarray] | np.ndarray:
    from numpy.lib.npyio import NpzFile
    f: NpzFile = np.load(filename)
    d = {int(k): v for k, v in f.items()}
    return d[n] if n is not None else d


def error(x, b: Biroot) -> float:
    return np.abs(b(x) - x ** (1 / b.n_))


def generate_matrix(b: Biroot,
                    x_linespace,
                    m_range: range,
                    err_func: Callable[[np.ndarray, Biroot], np.ndarray] = error,
                    clip: tuple[float, float] = None) -> np.ndarray:
    matrix: np.ndarray = np.zeros((len(m_range), len(x_linespace)))
    for i, m in enumerate(tqdm(m_range)):
        b.change_params(m=m)
        e = err_func(x_linespace, b)
        matrix[i, :] = np.clip(e, clip[0], clip[1]) if clip else e
    return matrix


def save_dataset() -> dict[str, np.ndarray]:
    """
    The Binomial Biroot:
    max_m = 200
    ns = (3, 4, 5, 6)
    x = np.linspace(0, 10_000, 10_000)

    The Gaussian Biroot:
    max_m = 70
    ns = (2, 3, 4, 5)
    x = np.linspace(0, 10_000, 10_000)

    The DAG Biroot:
    max_m = 70
    ns = (2, 3, 4, 5)
    x = np.linspace(0, 10_000, 10_000)
    dag = DAG().as_graph(basin=[5, 2, 7, 1, 8], depth=max_m, node_arity=3, func=lambda a, b, c: 1*a + 4*b + 3*c)
    b = Biroot(min(ns) + 1, min(ns), c=c, dag=dag)
    """
    max_m: int = 70
    ns = (2,3,4,5)  # for each root n=n_i, a different matrix is created
    c: int = 1
    x = np.linspace(0, 10000, 10000)

    matrices: dict[str, np.ndarray] = {}
    dag = DAG().as_graph(basin=[5, 2, 7, 1, 8], depth=max_m, node_arity=3, func=lambda a, b, c: 1*a + 4*b + 3*c)
    b: Biroot = Biroot(min(ns) + 1, min(ns), c=c, dag=dag)
    for n in ns:
        b.change_params(n=n)
        matrices[str(n)] = generate_matrix(b, x, range(n + 1, max_m))
    np.savez(f'dag_biroot_c_is_{c}.npz', **matrices)
    return matrices



if __name__ == '__main__':
    # Generate Dataset and save it.
    save_dataset()

    # Test loading a Dataset
    # d = load_dataset('binom_biroot_c_is_1.npz')
    # print(d)
