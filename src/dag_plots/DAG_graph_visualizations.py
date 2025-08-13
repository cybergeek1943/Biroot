import os

from sympy.categories import Diagram

os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'
from graphviz import Digraph
from dag import DAG, Level


def create_graphviz_from_DAG(dag_obj, node_arity=None, name="dag"):
    """
    Create Graphviz visualization from a DAG object

    Args:
        dag_obj: DAG object with levels already generated
        node_arity: Used for determining edge connections (if not provided, inferred)
        name: Name for the graph

    Returns:
        Graphviz Digraph object
    """
    dot = Digraph(name=name, comment='DAG Structure')
    dot.graph_attr['center'] = 'true'  # Add this line to center the graph
    dot.attr(rankdir='TB')  # Top to Bottom
    dot.attr('node', shape='circle', style='filled', fillcolor='lightblue', fontsize='12', width='0.5')
    dot.attr('edge', color='gray')


    # Get levels from DAG object
    levels = []
    for level_idx in sorted(dag_obj.levels.keys()):
        levels.append(list(dag_obj.levels[level_idx]))

    # If node_arity not provided, try to infer it
    if node_arity is None and len(levels) >= 2:
        # Infer from the growth pattern between first two levels
        if len(levels[0]) == 1 and len(levels[1]) == 1:
            node_arity = 1
        else:
            # Estimate based on level growth
            node_arity = max(2, len(levels[0]))  # Default to at least 2

    # Add nodes by level (ensures proper ranking)
    for level_idx, level in enumerate(levels):
        with dot.subgraph() as s:
            s.attr(rank='same')
            for node_idx, value in enumerate(level):
                node_id = f"L{level_idx}N{node_idx}"
                # Format the value nicely (handle floats vs ints)
                if isinstance(value, float) and value.is_integer():
                    label = str(int(value))
                else:
                    label = str(value)
                s.node(node_id, label)

    # Add edges based on node_arity
    if node_arity:
        for level_idx in range(1, len(levels)):
            for node_idx in range(len(levels[level_idx])):
                node_id = f"L{level_idx}N{node_idx}"

                # Connect to node_arity parents
                for j in range(node_arity):
                    parent_idx = node_idx - j
                    if 0 <= parent_idx < len(levels[level_idx - 1]):
                        parent_id = f"L{level_idx - 1}N{parent_idx}"
                        dot.edge(parent_id, node_id)

    return dot


def highlight_row(dot: Digraph, m: int, level: Level, color: str = 'coral'):
    for idx in range(len(level)):
        dot.node(f'L{m}N{idx}', style='filled', fillcolor=color)


def highlight_diagonal(dot: Digraph, m: int, level: Level, color: str = 'coral'):
    for idx in range(len(level)):
        dot.node(f'L{m-idx}N{idx}', style='filled', fillcolor=color)


if __name__ == '__main__':
    dag = DAG().as_graph(basin=[1, 2, 3], depth=3, node_arity=4)
    dot = create_graphviz_from_DAG(dag, 4)
    # highlight_row(dot, 3, level=dag.get_level(3))
    dot.render('dag_3', format='pdf', view=True)
