from biroot import Biroot
from typing import Callable, Any
import random as rd

type num = int | float
type LinearFunction = Callable


class Funcs:
    sum = lambda *args: sum([*args])
    lf_1 = lambda x1, x2, x3: 2 * (x1 + x2) + 3 * x3 - 10
    lf_2 = lambda x1, x2, x3: 3 * (x1 + 4 * x2 + 13 - x3) + - x1 + x3 - 19


class Level(list):
    """Level of a graph"""

    def __getitem__(self, idx: int | slice) -> int or 'Level':
        if type(idx) is slice:  # check if slice first
            return Level(super().__getitem__(idx))
        if idx < 0:
            return 0
        try:
            return super().__getitem__(idx)
        except IndexError:
            return 0

    def nodes_range(self, node_arity: int) -> range:
        """Returns the range for which a level nodes can be looped through."""
        return range(-node_arity + 1,
                     self.__len__())


class DAG:
    def __init__(self, basin: Level,
                 depth: int,
                 node_arity: int = 2,
                 func: LinearFunction = Funcs.sum,
                 step_caller: Callable[[Level], Any] = None,
                 save_levels: bool = True,
                 ) -> None:
        self.save_levels: bool = save_levels
        self.levels: list[Level] = []
        self.__dag(basin, depth, node_arity, func, step_caller)

    def regenerate(self, *args, **kwargs) -> None:
        self.levels.clear()
        self.__dag(*args, **kwargs)

    def __dag(self, l: Level,
              depth: int,
              node_arity: int = 2,
              func: LinearFunction = Funcs.sum,
              step_caller: Callable[[Level], Any] = None
              ) -> Level:
        """Recursive DAG generator."""
        if step_caller:
            step_caller(l)
        if depth == 0:
            self.levels.append(l)
            return l
        new_level = Level()
        for idx in l.nodes_range(node_arity):
            new_level.append(operator(l=l, idx=idx, node_arity=node_arity, func=func))
        if self.save_levels:
            self.levels.append(new_level)
        return self.__dag(l=new_level,
                          depth=depth - 1,
                          node_arity=node_arity,
                          func=func,
                          step_caller=step_caller)  # rd.choice([Funcs.lf_1, Funcs.sum, Funcs.lf_2]) works

    def getLevel(self, n: int = -1) -> Level:
        return self.levels[n]


def print_coefficients(l: Level) -> None:
    print(' '.join(map(str, l)))


def print_func(l: Level, k: int = 2) -> None:
    """Prints the function in a format that can be used in Geogebra"""
    print(Biroot.__creat_rational_def__(l, k=k))


def operator(l: Level, idx: int, node_arity: int, func: LinearFunction) -> num:
    """operator function that operates on two parent nodes"""
    return func(*[l[i] for i in range(idx, idx + node_arity)])  # map nodes to functions parameters


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
    start = Level([1])  # Basin of Attraction (first level)
    ll = DAG(start,
             depth=12,  # depth of recursion
             node_arity=4,  # max number of nodes to be used as arguments for the operator (n-ary graph as a result)
             func=Funcs.sum,  # to be used in the operator to create the next level of nodes based.
             step_caller=print_coefficients)
    print_func(ll.getLevel(5), k=2)
