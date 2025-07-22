from mathflow import Rational
from sympy import Symbol, Expr, Add
from sympy.core.numbers import E
from sympy import UnevaluatedExpr as uExpr
from sympy.abc import x
from typing import Callable, Self, Any
from math import ceil, exp
from dag import DAG, Level


class GaussianFunctions:
    class MachinePrec:
        f1: Callable[[int, float], float] = lambda m, x: exp(-2 * (x - m/2) ** 2 / m)  # mu = m/2
        f2: Callable[[int, float], float] = lambda m, x: exp(-2 * (x - m) ** 2 / m)  # mu = m
    f1: Callable[[int, float], Expr] = lambda m, x: uExpr(E) ** (-2 * (x - m/2) ** 2 / uExpr(m))  # mu = m/2
    f2: Callable[[int, float], Expr] = lambda m, x: uExpr(E) ** (-2 * (x - m) ** 2 / uExpr(m))  # mu = m
    f3: Callable[[int, float], Expr] = lambda m, x: uExpr(E) ** (-2 * (x - m/2) ** 2 / m)  # mu = m/2, evaluated index
    f4: Callable[[int, float], Expr] = lambda m, x: uExpr(E) ** (-2 * (x - m) ** 2 / m)  # mu = m, evaluated index


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
                 continuous: bool | Callable[[int, float], Expr | float] = False,
                 **kwargs) -> None:
        super().__init__()
        # IMPORTANT NOTE: Make sure that these attributes don't conflict with parent attributes
        self._m: int = m
        self._n: int = n
        self._c: float = c
        self._dag: DAG | Level | tuple[list[float], list[float]] = dag
        self._apply_upper_sum_bound_limit: bool = apply_upper_sum_bound_limit

        if continuous:
            self.set_continuous(continuous)
        # Create Function
        self._coeffs: tuple[list[float | Expr], list[float | Expr]] = ([], [])
        self.init_biroot_rational(**kwargs)

    def init_biroot_rational(self, **kwargs) -> Self:
        m, n, c = self._m, self._n, self._c
        numerator_coeffs: list[int | float] = []
        denominator_coeffs: list[int | float] = []
        if isinstance(self._dag, tuple):
            numerator_coeffs, denominator_coeffs = self._dag
        else:
            level: Level = self._dag.get_level(m) if isinstance(self._dag, DAG) else self._dag
            upper_bound: int = ceil(m / n) + (1 if m % n == 0 else 0) if self._apply_upper_sum_bound_limit else m  # 👉👉👉 TODO thoroughly document this math
            __mhl_kwargs: dict = {
                'heads': (0, 1),  # number and position of pointers
                'step': (n, n),  # the step for each pointer
                'end_pos': upper_bound  # how far the pointers should traverse
            }
            if kwargs: __mhl_kwargs.update(kwargs)
            for nc, dc in level.multihead_loop(**__mhl_kwargs):
                numerator_coeffs.append(nc)
                denominator_coeffs.append(dc)

        if c == 1:
            numerator: Expr = Add(*[coeff * x**k for k, coeff in enumerate(numerator_coeffs)])
            denominator: Expr = Add(*[coeff * x**k for k, coeff in enumerate(denominator_coeffs)])
        else:
            numerator: Expr = Add(*[coeff * c**(m - n*k) * x**k for k, coeff in enumerate(numerator_coeffs)])
            denominator: Expr = Add(*[coeff * c**(m - n*k - 1) * x**k for k, coeff in enumerate(denominator_coeffs)])

        self._coeffs = numerator_coeffs, denominator_coeffs
        self.set_expr(numerator / denominator)  # set the rationals internal expr
        return self

    def change_params(self, **kwargs) -> Self:
        for k, v in kwargs.items():  # update the biroot class attributes
            if hasattr(self, f'_{k}'):
                setattr(self, f'_{k}', kwargs.pop(k))
        return self.init_biroot_rational(**kwargs)

    def set_continuous(self, f: bool | Callable[[int, float], Expr | float], **kwargs) -> Self:
        if isinstance(f, bool) and f:
            f = GaussianFunctions.f1
        self._dag.as_continuous(f)
        return self.change_params(**kwargs)

    @property
    def numerator_coeffs(self) -> list[float | Expr]:
        return self._coeffs[0]

    @property
    def denominator_coeffs(self) -> list[float | Expr]:
        return self._coeffs[1]

    @property
    def biroot_params(self) -> dict[str, Any]:
        return {'m': self._m, 'n': self._n, 'c': self._c, 'dag': self._dag}

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
    b = Biroot(15, 3, c, continuous=True)
    b.print('latex')
