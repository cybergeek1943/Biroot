from typing import Callable, Any, Union, Sequence, Generator, Self, Iterator
from math import comb, ceil


def nCk(n: int, k: int) -> int:  # Alias
    if k < 0:
        return 0
    return comb(n, k)


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
                       repeat_step: int = -1) -> Generator[Sequence[int], None, None]:
        if len(heads) != len(step):
            raise IndexError('There must be the same number of `heads` as number of `step`')
        if repeat_step == -1:
            if self.f:
                raise ValueError('When level is a continuous function, `repeat` must be specified (other than -1).')
            else:
                repeat_step = ceil(len(self) / min(step)) - ceil(min(heads) / min(step))
        for _ in range(abs(repeat_step)):
            yield [self[h] for h in heads]
            heads = [h + s for h, s in zip(heads, step)]


class DAG:
    def __init__(self, as_continuous: Callable[[int, float], Any] | bool = None, **kwargs) -> None:
        self.levels: dict[int, Level] = {}
        if kwargs and 'basin' in kwargs and 'depth' in kwargs:
            self.as_graph(**kwargs)

        # Continuous Function Attributes
        self.f: Callable[[int, float], Any] | None = None
        self.continuous: bool = False
        if as_continuous:
            self.as_continuous(as_continuous)

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
                 node_arity: int | Sequence[int] = 2,  # also known as heads
                 step: int | Sequence[int] = 1,
                 repeat_step: int | Callable[[Level, Sequence[int], Sequence[int], int], int] = None,
                 save_levels: bool = True,
                 func: Callable[..., int | float | Sequence[int | float]] = Funcs.sum,
                 step_caller: Callable[[Level], Any] = None) -> Self:
        self.clear_graph()
        if not isinstance(basin, Level):
            basin: Level = Level(list(basin))
        if (len(basin) == 1 and
                basin[0] == 1 and
                node_arity == 2 and
                step == 1 and
                repeat_step is None and
                func == Funcs.sum):
            return self.__generate_pascals(depth, save_levels, step_caller)
        # Set up params for multihead loop
        self.levels[0] = basin
        if isinstance(node_arity, int):
            node_arity = [p for p in range(-node_arity + 1, 1)]
        if isinstance(step, int):
            step = [step for _ in range(len(node_arity))]
        if repeat_step is None:
            _min_head: int = min(node_arity)
            _min_step: int = min(step)
            repeat_step = lambda level, heads, step, level_idx: ceil(len(level) / _min_step) - ceil(_min_head / _min_step)
        elif isinstance(repeat_step, int):
            _repeat_step: int = repeat_step
            repeat_step = lambda *args: _repeat_step
        self.__generate_graph(basin, depth, node_arity, step, repeat_step, save_levels, func, step_caller)
        return self

    def __generate_graph(self, l: Level,
                         depth: int,
                         heads: Sequence[int],
                         step: Sequence[int],
                         repeat_step: Callable[[Level, Sequence[int], Sequence[int], int], int],
                         save_levels: bool,
                         func: Callable[..., int | float | Sequence[int | float]],
                         step_caller: Callable[[Level], Any],
                         __level_idx: int = 0) -> Self:
        """Recursive DAG generator."""
        if step_caller:
            step_caller(l)
        if __level_idx == depth:
            if not save_levels: self.levels[__level_idx] = l
            return self
        new_level: Level = Level()
        for args in l.multihead_loop(heads, step, repeat_step=repeat_step(l, heads, step, __level_idx)):
            nv: int | float | Sequence[int | float] = func(*args)
            if isinstance(nv, Sequence):
                new_level.extend(nv)
            else:
                new_level.append(nv)
        # NOTE: this is old code with less flexibility as multihead_loop():
        # for idx in range(-node_arity + 1, len(l)):
        #     new_level.append(func(*[l[i] for i in range(idx, idx + node_arity)]))
        if save_levels:
            self.levels[__level_idx + 1] = new_level

        # Recursive Call
        return self.__generate_graph(new_level, depth, heads, step, repeat_step, save_levels, func, step_caller, __level_idx + 1)

    def __generate_pascals(self, depth: int, save_levels: bool, step_caller: Callable[[Level], Any]) -> Self:
        """Optimization in the event the Basin=[1]"""
        if save_levels:
            for n in range(depth):
                self.levels[n] = Level([nCk(n, k) for k in range(n + 1)])
                if step_caller:
                    step_caller(self.levels[-1])
        self.levels[depth] = Level([nCk(depth, k) for k in range(depth+1)])
        return self

    def set_level_kwargs(self, **kwargs) -> Self:
        for l in self.levels.values():
            for k, v in kwargs.items():
                setattr(l, k, v)
        return self

    def get_diagonal(self, start_idx: int,
                     x_step: Callable[[int], int] | int = 1,
                     y_step: Callable[[int], int] | int = -1,
                     repeat_step: int = None) -> Level:
        """
        Useful if we want to get a diagonal to analyze the coeffs of Chebyshev Polynomials, Fibonacci polynomials, etc.
        """
        if isinstance(x_step, int):
            _x = x_step
            x_step = lambda x: x + _x
        if isinstance(y_step, int):
            _y = y_step
            y_step = lambda y: y + _y
        out: list[int | float] = []
        x, y = 0, start_idx  # current positions

        if repeat_step:
            for _ in range(repeat_step):
                out.append(self.get_level(y, default=0)[x])
                x = x_step(x)
                y = y_step(y)
        else:
            while y != -1 and (v:=(self.get_level(y, default=0)[x])) != 0:
                out.append(v)
                x = x_step(x)
                y = y_step(y)
        return Level(out)

    def get_level(self, m: int, default: Level | Any = ...) -> Level:
        if m == -1:
            if not self.levels:
                raise IndexError('No levels exist in empty DAG')
            if self.continuous and self.f:
                raise IndexError('Cannot determine "last" level for continuous functions')
            m = max(self.levels.keys())
        try:
            return self.levels[m]
        except KeyError:
            if default is ...:
                if self.continuous and self.f:
                    default: Level = Level((lambda x: self.f(m, x)))
                else:
                    default: Level = Level([nCk(m, k) for k in range(m + 1)])
            elif not isinstance(default, Level):
                default: Level = Level(obv=default)
            return default

    def __getitem__(self, m: int) -> Level:  # enables getting row and column by DAG[r][c]
        return self.get_level(m)

    def __str__(self) -> str:
        return '\n'.join([f'{i}: {l}' for i, l in self.levels.items()])

    def __repr__(self) -> str:
        return str(self)

    def __iter__(self) -> Iterator[Level]:
        return self.levels.values().__iter__()

    def __len__(self) -> int:
        return len(self.levels)


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
    # g = DAG()
    # print(g[5][3])
    dag = DAG().as_graph(basin=[1], depth=10, func=lambda x, y: x + y)
    print(dag)
    print('\n====\n')
    for r in range(20):
        print(sum(l:=dag.get_diagonal(r)), l)

    # l = Level([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # for i, j, k in l.multihead_loop((-2, -1, 0), step=(2, 2, 2), repeat_step=10 // 2 + 1):
    #     print(i, j, k)
