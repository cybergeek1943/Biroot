from mathflow import Rational
from mpmath import levin
from sympy import Symbol, Expr, Add
from sympy.core.numbers import E
from sympy import UnevaluatedExpr as uExpr
from sympy.abc import x
from typing import Callable, Self, Any
from math import ceil, exp
from dag import DAG, Level


class GaussianFunctions:
    class MachinePrec:
        f1: Callable[[float], float] = lambda m, x: exp(-2 * ((x - m/2)**2) / m)  # mu = m/2
        f2: Callable[[float], float] = lambda m, x: exp(-2 * (x - m) ** 2 / m)  # mu = m
    f1: Callable[[float], Expr] = lambda m, x: uExpr(E) ** (-2 * (x - m/2) ** 2 / uExpr(m))  # mu = m/2
    f2: Callable[[float], Expr] = lambda m, x: uExpr(E) ** (-2 * (x - m) ** 2 / uExpr(m))  # mu = m
    f3: Callable[[float], Expr] = lambda m, x: uExpr(E) ** (-2 * (x - m/2) ** 2 / m)  # mu = m/2, evaluated index
    f4: Callable[[float], Expr] = lambda m, x: uExpr(E) ** (-2 * (x - m) ** 2 / m)  # mu = m, evaluated index


class Biroot(Rational):
    """
    Implements the biroot approximation for ith roots. It uses the nth row version of the formula rather than the initial (n*i)th row formula.
    It allows use to set a default c centering or, alternatively, use a parameterized c centering.

    PROMISING RESULT: if i=4 and denominator_offset=1, seems to be a better function to approximate the next c.
    """
    def __init__(self,
                 m: int,
                 n: int = 2,
                 c: float | Symbol = 1,
                 dag: DAG | Level | tuple[list[float], list[float]] = DAG(),
                 apply_upper_sum_bound_limit: bool = True,
                 continuous: bool | Callable[[float], Expr | float] = False,
                 **kwargs) -> None:
        super().__init__()
        # IMPORTANT NOTE: Make sure that these attributes don't conflict with parent attributes
        self.m_: int = m
        self.n_: int = n
        self.c_: float = c
        self.dag_: DAG | Level | tuple[list[float], list[float]] = dag
        self.apply_upper_sum_bound_limit_: bool = apply_upper_sum_bound_limit


        # Create Function
        self.coeffs_: tuple[list[float | Expr], list[float | Expr]] = ([], [])
        self.combined_coeffs_: list[float | Expr] = []
        if continuous:
            self.set_continuous(continuous)
            return
        self.init_biroot_rational(**kwargs)

    def init_biroot_rational(self, **kwargs) -> Self:
        m, n, c = self.m_, self.n_, self.c_
        numerator_coeffs: list[int | float] = []
        denominator_coeffs: list[int | float] = []
        level: Level | None = None
        if isinstance(self.dag_, tuple):
            numerator_coeffs, denominator_coeffs = self.dag_
        else:
            if isinstance(self.dag_, DAG):
                level: Level = self.dag_.get_level(m)
            else:  # if dag is a Level
                level: Level = self.dag_

            if not level.f:  # if not continuous
                m = len(level)
            upper_bound: int = ceil(m / n) if self.apply_upper_sum_bound_limit_ else m
            __mhl_kwargs: dict = {
                'heads': (0, 1),  # number and position of pointers
                'step': (n, n),  # the step for each pointer
                'repeat_step': upper_bound  # how far the pointers should traverse
            }
            if kwargs: __mhl_kwargs.update(kwargs)
            for nc, dc in level.multihead_loop(**__mhl_kwargs):
                numerator_coeffs.append(nc)
                denominator_coeffs.append(dc)
                self.combined_coeffs_.extend((nc, dc))
            m -= 1  # for the sake of the c parameter.

        if c == 1:
            numerator: Expr = Add(*[coeff * x**k for k, coeff in enumerate(numerator_coeffs)])
            denominator: Expr = Add(*[coeff * x**k for k, coeff in enumerate(denominator_coeffs)])
        else:
            numerator: Expr = Add(*[coeff * c**(m - n*k) * x**k for k, coeff in enumerate(numerator_coeffs)])
            denominator: Expr = Add(*[coeff * c**(m - n*k - 1) * x**k for k, coeff in enumerate(denominator_coeffs)])

        self.coeffs_ = numerator_coeffs, denominator_coeffs
        self.set_expr(numerator / denominator)  # set the rationals internal expr
        return self

    def change_params(self, **kwargs) -> Self:
        for k, v in tuple(kwargs.items()):  # update the biroot class attributes
            if hasattr(self, f'{k}_'):
                setattr(self, f'{k}_', kwargs.pop(k))
        return self.init_biroot_rational(**kwargs)

    def set_continuous(self, f: bool | Callable[[float], Expr | float], **kwargs) -> Self:
        if isinstance(f, bool) and f:
            f = GaussianFunctions.f1
        self.dag_.as_continuous(f)
        return self.change_params(**kwargs)

    @property
    def numerator_coeffs(self) -> list[float | Expr]:
        return self.coeffs_[0]

    @property
    def denominator_coeffs(self) -> list[float | Expr]:
        return self.coeffs_[1]

    @property
    def combined_coeffs(self) -> list[float | Expr]:
        return self.combined_coeffs_

    @property
    def biroot_params(self) -> dict[str, Any]:
        return {'m': self.m_, 'n': self.n_, 'c': self.c_, 'dag': self.dag_}

    def convert_sci_notation_in_str(self) -> str:
        return super().__str__().replace('e-', '*10**-')


def RecursiveBiroot(b: Biroot, x, c, steps: int | float = 0.001):
    """
    - NOTE: currently the dynamic recursion is unreliable because the way the accuracy threshold is detected. For this reason, is better to specify the exact number of steps. If a float for steps is used, you may get an infinite loop because the accuracy threshold is too extreme.
    - NOTE for `i==2`: `i` needs to be odd so that it levels out. If you don't do this, you can get an Overload error.
    - NOTE for `i > 2`: to get accurate results, `n-1` must be evenly divisible by `i`.
    """
    if isinstance(steps, float):
        accuracy: float = steps
        t: float = 0
        while abs(t - c) > accuracy:
            t = c
            c = b(x, c)
        return c
    for _ in range(steps):
        c = b(x, c)
    return c


if __name__ == "__main__":
    from sympy.abc import c

    # f = lambda m, x: exp(-2 * (x - m/2) ** 2 / m)  # mean = m/2 and standard_dev = sqrt(m/4)
    # dag = DAG().as_continuous(f)
    # dag = DAG().as_graph(basin=[1, 3, 2], depth=40, node_arity=3)
    # b = Biroot(-1, 2, dag=dag.get_diagonal(40, 2, -2))
    # print(b.print.latex(order='old'))

    b = Biroot(m=12, n=3, c=c)

    b.print('latex')

    # ======== Randomized DAG Biroot Analysis ========
    # import random as rd
    # basin = [rd.randint(1, 9) for _ in range(rd.randint(1, 8))]
    # node_arity = rd.randint(2, 4)
    # random_func_coeffs = [rd.randint(1, 9) for _ in range(node_arity)]
    # random_func = lambda *args: sum([args[i]*random_func_coeffs[i] for i in range(len(args))])
    # dag = DAG().as_graph(basin=basin, depth=40, node_arity=node_arity, func=random_func)
    # b = Biroot(-1, 3, dag=dag)  # dag.get_diagonal(40)
    # print(b.print.latex(order='old'))
    # print(basin)
    # print(node_arity)
    # print(random_func_coeffs)
    # import numpy as np
    # def error(x, b: Biroot) -> float:
    #     return np.abs(b(x) - x ** (1 / b.n_))
    # x = np.linspace(0, 10_000, 10_000)
    # y = error(x, b)
    # mu = np.mean(y)
    # sigma = np.std(y)
    # print(mu, sigma)
    #
    # formatted_row = (  # Latex row for table
    #     f"${basin}$ & "
    #     f"{node_arity} & "
    #     f"${random_func_coeffs}$ & "
    #     f"{mu:.1e} & "
    #     f"{sigma:.4f}\\\\"
    # )
    #
    # print(formatted_row)


    # ======== Continuous Biroot Analysis ========
    # b: Biroot = Biroot(18, 3, c=1, apply_upper_sum_bound_limit=False, continuous=True)
    # print(b.print.latex())
    # import numpy as np
    # def error(x, b: Biroot) -> float:
    #     return np.abs(b(x) - x ** (1 / b.n_))
    # x = np.linspace(0, 100_000_000, 10_000_000)
    # y = error(x, b)
    # print(y)
    # print(np.mean(y))
    # print(np.std(y))

    # ======== Optimal Condition ========
    # b = Biroot(3, 6)
    #
    # b.print('latex', order='old')
    # o_i: list[int] = []
    # for m in range(3, 78):
    #     b.change_params(m=m)
    #     v = b.subs(x, 1)
    #     if v.equals(1):
    #         o_i.append(m)
    # print(o_i)
    # print(len(o_i))

    # sixth_root = [7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73]
    # fifth_root = [6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61]
    # fourth_root = [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]
    # cube_root = [4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37]
    # for i, l in enumerate((cube_root, fourth_root, fifth_root, sixth_root)):
    #     print(f'{3+i} &', ' & '.join(map(str, l)), r'\\')
