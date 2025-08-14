# 🔢Biroot: Nth Root Approximation Method
[![DOI](https://zenodo.org/badge/1022357951.svg)](https://doi.org/10.5281/zenodo.16878055)

Python library for computing nth roots using directed acyclic graphs (DAGs) or Gaussian functions. Based on the paper (draft in-progress) *DAG-Gaussian Encoding for nth Root Function Approximation*

## ✨ Features

- **Biroot Class**: Root approximation using rational functions
- **Continuous Functions**: Gaussian function interpolation on the DAG structures
- **DAG Structure**: Pascal's triangle and custom linear operators 
- **Recursive Algorithms**: High-precision root computation

## 🚀 Quick Start

Biroot Expressions
```python
from biroot import Biroot

# Square root approximation, 8th row of DAG
b = Biroot(m=8)  # you can specify the DAG with `dag=DAG()`
b.print('latex')  # print the function to terminal
denom_coeffs = b.denominator_coeffs  # analyze the coeffs

result = b(9)  # will be approximately 3
```
 
```python
# Cube root approximation, Gaussian function with m=8, parameterize c
b = Biroot(m=8, n=3, c=Symbol('c'), continuous=True)
b.print('latex')

result = b(8, 2)  # cube root of 8, centered at 2^3

# Because there is full MathFlow support, we get more analysis tools
roots = b.all_roots()
b_prime = b.diff()  # find the derivative of b
```

DAG Data Structure
```python
from dag import DAG

# Pascal's triangle generation (default)
dag = DAG()
print(dag[5])  # [1, 5, 10, 10, 5, 1]

# Multi-node random basin with ternary operations on the nodes
dag = DAG(basin=[3, 6, 2, 8], depth=30, node_arity=3)
print(dag[3])  # [3, 15, 38, 71, 96, 107, 89, 60, 26, 8]
```

## 🧮 Mathematical Foundation

Explores attractor properties of binomial coefficient patterns in iterative root approximation, potentially explaining Newton's method and Padé binomial structures. Based on the work in the paper *DAG-Gaussian Encoding for nth Root Function Approximation*.

## 🪜Next Steps

- Write extensive unit tests
- Add more specialized analysis methods
- Make a series of Jupyter Notebooks

## 📦 Dependencies

`mathflow`, standard library modules
