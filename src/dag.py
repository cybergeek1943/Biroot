from typing import Callable, Any, Union, Sequence, Generator, Self
from math import e, comb
import random as rd


def nCk(n: int, k: int) -> int:  # Alias
    if k < 0:
        return 0
    return comb(n, k)


type LinearFunction = Callable[[float], float]
class Funcs:
    sum = lambda *args: sum([*args])
    lf_1 = lambda x1, x2, x3: 2 * (x1 + x2) + 3 * x3 - 10
    lf_2 = lambda x1, x2, x3: 3 * (x1 + 4 * x2 + 13 - x3) + - x1 + x3 - 19


class Level(list):
    """Level of a graph"""
    def __init__(self, f: list[int | float] | Callable[[float], float] = None,
                 obv: Any = 0,
                 domain: tuple = None) -> None:
        super().__init__(f if isinstance(f, list) else [])
        self.f: Callable[[float], Any] | None = f if callable(f) else None
        self.obv: Any = obv  # outside bounds value
        self.domain: tuple[float, float] | None = domain

    def __call__(self, x: float | int):
        if self.domain and not self.domain[0] <= x <= self.domain[1]:
            return self.obv
        try:
            return self.f(x)
        except TypeError:
            return self[x]

    def __getitem__(self, idx: int | slice | float) -> Union[int, float, 'Level']:
        if type(idx) is slice:  # check if slice first
            return Level(super().__getitem__(idx))
        if self.domain and not self.domain[0] <= idx <= self.domain[1]:
            return self.obv
        if self.f:
            return self.f(idx)
        if idx < 0:
            return self.obv
        try:
            return super().__getitem__(idx)
        except IndexError:
            return self.obv

    def multihead_loop(self, heads: Sequence[int] = (0, 1),
                       step: Sequence[int] = (1, 1),
                       start_pos: int = 0, end_pos: int = -1) -> Generator[Sequence[int], None, None]:
        if len(heads) != len(step):
            raise IndexError('There must be the same number of `heads` as number of `steps`')
        if end_pos == -1:
            if self.f:
                raise ValueError('When level is a continuous function, end_pos must be specified (other than -1).')
            else:
                end_pos = len(self)
        if start_pos != 0:
            heads = [h + start_pos for h in heads]
        for _ in range(start_pos, end_pos):
            yield [self[h] for h in heads]
            heads = [h + s for h, s in zip(heads, step)]


class DAG:
    def __init__(self, basin: Level | Sequence[int | float] = None,
                 depth: int = None,
                 node_arity: int = 2,
                 save_levels: bool = True,
                 func: LinearFunction = Funcs.sum,
                 step_caller: Callable[[Level], Any] = None) -> None:
        self.levels: dict[int, Level] = {}
        if basin and depth:
            self.as_graph(basin, depth, node_arity, save_levels, func, step_caller)

        # Continuous Function Attributes
        self.f: Callable[[int, float], Any] | None = None
        self.continuous: bool = False

    def clear_graph(self):
        self.levels.clear()

    def as_continuous(self, continuous: Callable[[int, float], Any] | bool) -> Self:
        """If `continuous` is a callable, its signature must be in form (m: int, x: float) where m is the row specifier and x is the free variable."""
        if isinstance(continuous, bool):
            self.continuous = continuous
        elif callable(continuous):
            self.f = continuous
            self.continuous = True
        return self

    def as_graph(self, basin: Level | Sequence[int | float],
                 depth: int,
                 node_arity: int = 2,
                 save_levels: bool = True,
                 func: LinearFunction = Funcs.sum,
                 step_caller: Callable[[Level], Any] = None) -> Self:
        self.clear_graph()
        if not isinstance(basin, Level):
            basin: Level = Level(list(basin))
        if len(basin) == 1 and basin[0] == 1 and node_arity == 2 and func == Funcs.sum:
            return self.__generate_pascals(depth, save_levels, step_caller)
        self.levels[0] = basin
        return self.__generate_graph(basin, depth, node_arity, save_levels, func, step_caller)

    def __generate_graph(self, l: Level,
                         depth: int,
                         node_arity: int,
                         save_levels: bool,
                         func: LinearFunction,
                         step_caller: Callable[[Level], Any],
                         __level_idx: int = 0) -> Self:
        """Recursive DAG generator."""
        if step_caller:
            step_caller(l)
        if depth == __level_idx:
            if not save_levels: self.levels[__level_idx] = l
            return self
        new_level = Level()
        for idx in range(-node_arity + 1, len(l)):
            new_level.append(func(*[l[i] for i in range(idx, idx + node_arity)]))
        if save_levels:
            self.levels[__level_idx + 1] = new_level
        return self.__generate_graph(new_level, depth, node_arity, save_levels, func, step_caller, __level_idx + 1)  # rd.choice([Funcs.lf_1, Funcs.sum, Funcs.lf_2]) works

    def __generate_pascals(self, depth: int, save_levels: bool, step_caller: Callable[[Level], Any]) -> Self:
        """Optimization in the event the Basin=[1]"""
        if save_levels:
            for n in range(depth):
                self.levels[n] = Level([nCk(n, k) for k in range(n + 1)])
                if step_caller:
                    step_caller(self.levels[-1])
        self.levels[depth] = Level([nCk(depth, k) for k in range(depth+1)])
        return self

    def get_level(self, m: int) -> Level:
        return self.levels.get(m,
                               Level(lambda x: self.f(m, x)) if self.continuous and self.f else
                               Level([nCk(m, k) for k in range(m+1)]))

    def __getitem__(self, m: int) -> Level:  # enables getting row and column by DAG[r][c]
        return self.get_level(m)

    def __str__(self) -> str:
        return '\n'.join([f'{i}: {l}' for i, l in self.levels.items()])

    def __repr__(self) -> str:
        return str(self)


# TODO Current Findings:
# We can start with N number of random nodes as the first level.
# We can use any linear operator function we like.
# We can use any arity of nodes we like for the operator.
# We can change the arity of nodes at each level (apparently randomly at each level).
# We can (sometimes/sort of) change the arity of nodes at each index of a node in a level (works best by increasing by the index of current node: node_arity += idx).
# We can change the operator function at each level (apparently randomly at each level).


# TODO Negative Results:
# What happens if we skip the operator on every ith node?
# What happens if we use a different operator at each idx of a node when creating the next level?


if __name__ == '__main__':
    # The graph exhibits an Attractor Property (and Structural Invariance) to the nth root function.
    # It may explain why Newton's method, when iterated symbolically, yields the observed binomial patterns - both are manifestations of the same underlying attractor.
    # start = Level([1])  # Basin of Attraction (first level)
    # ll = DAG(start,
    #          depth=12,  # depth of recursion
    #          node_arity=4,  # max number of nodes to be used as arguments for the operator (n-ary graph as a result)
    #          func=Funcs.sum,  # to be used in the operator to create the next level of nodes based.
    #          step_caller=print_coefficients)
    # print_func(ll.getLevel(5), k=2)
    g = DAG()
    print(g[5][3])
