from typing import Callable
from math import comb, ceil
from abc import ABC, abstractmethod


def nCk(n: int, k: int) -> int:
    if k < 0:
        return 0
    return comb(n, k)


class DynamicFunction(ABC):
    """An ABC class that defines a Dynamic Function."""
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class Function(DynamicFunction):  # Has 1 def
    def __call__(self, *args, **kwargs):
        pass


class RationalFunction(DynamicFunction):  # Has 2 defs for numerator and denominator
    def __call__(self, *args, **kwargs):
        pass


class Biroot(RationalFunction):
    """
    Implements the iterative approximation for ith roots. It uses the nth row version of the formula rather than the initial (n*i)th row formula.
    It allows use to set a default c centering or, alternatively, use a parameterized c centering.

    PROMISING RESULT: if i=4 and denominator_offset=1, seems to be a better function to approximate the next c.
    """

    def __init_callable(self) -> None:
        self.f_def = self.create_ith_root_def()
        self.callable = eval(f"lambda x{', c' if self.parameterize_c else ''}: {self.f_def}")

    def __setattr__(self, key, value):
        if key in self.__dict__ and key in ('n', 'c', 'i', 'parameterize_c', 'inverse', 'numerator_offset', 'denominator_offset'):
            super().__setattr__(key, value)
            self.__init_callable()
        else:
            super().__setattr__(key, value)

    def __init__(self,
                 n: int = 8,
                 c: float = 1,
                 i: int = 2,
                 parameterize_c: bool = False,
                 inverse: bool = False,
                 ) -> None:
        """
        :param n: Thought of as the row
        :param c: Centered around c^i
        :param i: The index of the root
        :param parameterize_c: Whether to make c a bound variable parameter
        :param inverse: Whether to make the biroot an inverse root (useful for RecursiveBiroot)
        """
        # Parameters
        self.n: int = n
        self.c: float = c
        self.i: int = i
        self.parameterize_c: bool = parameterize_c
        self.inverse: bool = inverse
        # offset can be used, for example, to choose every other element in a row while skipping elements in between a choice. Example, i=4 and offset=1 (1+1=2 so offset is actually two in the denominator)
        self.numerator_offset: int = 0
        self.denominator_offset: int = 0

        # Internals
        self.f_def: str = ''
        self.callable: Callable = lambda: None
        self.__init_callable()

    def __call__(self, x: float, c: float = 1) -> float:
        if not self.callable:
            raise ReferenceError('Function has not been created yet.')
        if self.parameterize_c:
            return self.callable(x, c)
        return self.callable(x)

    def __str__(self):
        return self.export_formated_str()

    def pprint(self) -> None:
        print(self.export_formated_str(True))

    @property
    def total_operations_count(self) -> int:
        s: str = self.f_def
        s = s.replace('**', '*')
        s = s.replace('+', '*')
        s = s.replace('/', '*')
        return s.count('*')

    def export_formated_str(self, pretty: bool = None) -> str:
        s: str = self.f_def
        s = s.replace('+', ' + ')
        s = s.replace('**', '^')
        s = s.replace('*', ' ')
        if pretty:
            n, d = s[1:-1].split(')/(')
            max_len: int = max(len(n), len(d))
            s = f'{n:^{max_len}}\n{'-' * max_len}\n{d:^{max_len}}'
        return s

    def export_expression_str(self) -> str:
        return self.f_def

    @staticmethod
    def str_from_numerator_denominator(numerator: list[str], denominator: list[str]) -> str:
        # if numerator[-1][0] == '0':
        #     numerator.pop(-1)  # because of the extra 0 term that sometimes occurs.
        s: str = f"({'+'.join(numerator)})/({'+'.join(denominator)})"
        s = s.replace('*c**0', '')
        s = s.replace('c**1*', 'c*')
        s = s.replace('*x**0', '')
        s = s.replace('x**1+', 'x+')
        return s

    def create_ith_root_def(self) -> str:
        n, c, i, parameterize_c, no, do = self.n, self.c, self.i, self.parameterize_c, self.numerator_offset, self.denominator_offset
        # NOTE: no and do (offset for numerator and denominator) is just for doing specific tests and is not part of the actual formula.
        nr: range = range(0, ceil(n / i) + (1 if n % i == 0 else 0) - no)
        dr: range = range(0, ceil(n / i) - do)
        if parameterize_c:
            numerator: list[str] = [f'{nCk(n, i*k+no)}*c**{n-i*k-no}*x**{k}' for k in nr]
            denominator: list[str] = [f'{nCk(n, i*k+1+do)}*c**{n-i*k-1-do}*x**{k}' for k in dr]
        elif c != 1:
            numerator: list[str] = [f'{nCk(n, i*k+no) *c**(n-i*k-no)}*x**{k}' for k in nr]
            denominator: list[str] = [f'{nCk(n, i*k+1+do) *c**(n-i*k-1-do)}*x**{k}' for k in dr]
        else:
            numerator: list[str] = [f'{nCk(n, i*k+no)}*x**{k}' for k in nr]
            denominator: list[str] = [f'{nCk(n, i*k+1+do)}*x**{k}' for k in dr]
        return self.str_from_numerator_denominator(
            numerator=denominator,
            denominator=numerator
        ) if self.inverse else self.str_from_numerator_denominator(
            numerator=numerator,
            denominator=denominator)

    # def __create_ith_root_def(self) -> str:  # FOR EXPERIMENTAL TESTS
    #     n = self.n
    #     r1: range = range(0, ceil(n / 2)+1)
    #     r2: range = range(0, ceil(n / 2))
    #     numerator: list[str] = [f'{nCk(n, 2 * k)}*x**{k}' for k in r1]
    #     denominator: list[str] = [f'{nCk(n, 2 * k + 1)}*x**{k}' for k in r2]
    #     return self.str_from_numerator_denominator(numerator, denominator)

    @staticmethod  # TODO make a dedicated class for constructing rational functions from coefficients
    def __creat_rational_def__(coefficients: list[int], inverse: bool = False, k: int = 2) -> str:
        """Just for creating other rational functions"""
        numerator_coefficients = coefficients[::k]
        denominator_coefficients = coefficients[1::k]
        c_indices = [i for i in range(len(coefficients))]
        c_indices.reverse()
        numerator: list[str] = [f'{coefficient}*x**{i}'
                                for coefficient, i in zip(numerator_coefficients,
                                                          range(len(numerator_coefficients)))]
        denominator: list[str] = [f'{coefficient}*x**{i}'
                                  for coefficient, i in zip(denominator_coefficients,
                                                            range(len(denominator_coefficients)))]
        return Biroot.str_from_numerator_denominator(
            numerator=denominator,
            denominator=numerator
        ) if inverse else Biroot.str_from_numerator_denominator(
            numerator=numerator,
            denominator=denominator)


# TODO look into using inverse square root to avoid overflow errors.
class RecursiveBiroot(Biroot):
    def __init__(self,
                 steps: float | int,
                 n: int = 8,
                 c: float = 1,
                 i: int = 2,
                 ) -> None:
        """
        - NOTE: currently the dynamic recursion is unreliable because the way the accuracy threshold is detected. For this reason, is better to specify the exact number of steps. If a float for steps is used, you may get an infinite loop because the accuracy threshold is too extreme.
        - NOTE for `i==2`: `i` needs to be odd so that it levels out. If you don't do this, you can get an Overload error.
        - NOTE for `i > 2`: to get accurate results, `n-1` must be evenly divisible by `i`.

        :param n: If a float is passed it will be used to reach that level of accuracy. If it's an integer it will do n many recursive steps.
        :param c: The initial guess (centered around c^i).
        :param i: The index of the root.
        """
        super().__init__(
            n=n,
            c=c,
            i=i,
            parameterize_c=True,
            inverse=True
        )
        self.steps: float | int = steps

        # Runtime info
        self.recursive_steps_count: int = 0  # number of recursive steps in last call.

    def __call__(self, x: float, c: float | None = None):
        """Performs a recursive Biroot calculation on c... basically in the form B(x, B(x, B(x, c)))."""
        if not c: c = self.c
        self.recursive_steps_count = 0
        B: Callable = super().__call__
        if isinstance(self.steps, float):
            accuracy: float = self.steps
            t: float = 0
            while abs(t - c) > accuracy:
                t = c
                c = 1 / B(x, c)
                self.recursive_steps_count += 1
            return c
        for _ in range(self.steps):
            c = 1 / B(x, c)
            self.recursive_steps_count += 1
        return c

    @property
    def total_operations_count(self):
        return self.recursive_steps_count * super().total_operations_count


class NewtonRaphsonRoot(DynamicFunction):
    def __init__(self,
                 steps: float | int = 0.000001,
                 c: float = 1,
                 i: int = 2,
                 ) -> None:
        """
        :param steps: If a float is passed it will be used to reach that level of accuracy. If it's an integer it will do n many recursive steps.
        :param c: The initial guess (centered around c^i)
        :param i: The index of the root
        """
        # Parameters
        self.steps: float | int = steps
        self.c: float = c
        self.i: int = i

        # Runtime info
        self.recursive_steps_count: int = 0  # number of recursive steps in last call.

    def __call__(self, x: float, c: float | None = None) -> float:
        if not c: c = self.c
        return self._nr_method(lambda p: p**self.i-x, lambda p: self.i*p**(self.i-1), c)

    @property
    def total_operations_count(self):
        return self.recursive_steps_count * 7  # a total of 7 operation per recursive step

    def _nr_method(self, f: callable, fd: callable, c: float) -> float:
        """Generic implementation of the newtons-raphson method.
        f is the function and fd is the functions derivative. c is the initial guess."""
        self.recursive_steps_count = 1  # 1 to count the first assignment at x2
        x1: float = c
        x2: float = x1 - (f(x1) / fd(x1))
        if isinstance(self.steps, float):
            accuracy: float = self.steps
            while abs(x2 - x1) > accuracy:
                x1 = x2
                x2 = x2 - (f(x2) / fd(x2))
                self.recursive_steps_count += 1
            return x2
        for _ in range(self.steps):
            x2 = x2 - (f(x2) / fd(x2))
            self.recursive_steps_count += 1
        return x2


def find_error_boundaries(biroot_: Biroot, error_threshold: float, initial_guesses: tuple | list) -> list:
    """Find the exact boundaries where error equals the threshold."""
    error_func = lambda x: abs(biroot_(x) - x ** (1 / biroot_.i)) - error_threshold
    def derivative(x, h=0.00001):  # Derivative approximation
        return (error_func(x + h) - error_func(x)) / h
    boundaries: list[float] = []
    for guess in initial_guesses:
        x: float = guess
        for _ in range(100):  # Max iterations
            fx = error_func(x)
            if abs(fx) < 0.00001:  # Convergence threshold
                boundaries.append(x)
                break
            dfx = derivative(x)
            x = x - fx / dfx
    return sorted(boundaries)


# TODO 👉👉👉 Show that B(c, c^2) hold for all fixed points less than c... this is to show that sequences of rational functions maintain all previous interpolation points.


if __name__ == "__main__":
    # Error Bounds
    # b = Biroot(n=16, i=2, c=4)
    # err = find_error_boundaries(b, error_threshold=0.00001, initial_guesses=[0.1, 40])
    # print('err bounds:', err)
    # v = 20  # value
    # print('biroot:', b(v))  # biroot
    # print('actual:', v**(1/2))  # actual

    b = Biroot(n=12, i=2, parameterize_c=True)
    print(b)
    print(b(57))
    print(57**(1/2))
    print(find_error_boundaries(b, 0.00001, [0.1, 20]))
    print('\n\n')  # add some space

    # --------------------------- Example ---------------------------
    # # ================ Example Biroot ================
    # biroot = Biroot(n=8, i=2, parameterize_c=True)
    # print(biroot)
    # print(biroot(4))
    # print('\n\n')  # add some space
    #
    #
    # # ================ Example Recursive Algorithms (cube root) ================
    # num = 9898988980029898989898  # input number
    # n_r = NewtonRaphsonRoot(i=3)
    # print(n_r(num))
    # print(n_r.recursive_steps_count)
    # print('\n\n')  # add some space
    # r_br = RecursiveBiroot(steps=20, n=8, i=3)  # 37th row of pascal's triangle. 7 recursive steps.
    # print(r_br(num))
    # print(r_br.recursive_steps_count)


    # ================ Output ================
    # (1*c**8+28*c**6*x+70*c**4*x**2+28*c**2*x**3+1*x**4)/(8*c**7+56*c**5*x+56*c**3*x**2+8*c*x**3)
    # 2.000609756097561
    #
    # 21471560.72538337
    # 87
    #
    # 21471560.725383375
    # 7
    i=6
    f = Biroot(n=1, i=i, parameterize_c=True)
    for n in range(50):
        print(f.n, f(2**i, 2))
        f.n += 1
