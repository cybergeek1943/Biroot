"""
Empirical Convergence Rate Analysis via Log-Log Regression

This module provides tools for analyzing the empirical convergence rates of numerical
methods using log-log regression to identify power law relationships in error decay.

MATHEMATICAL FOUNDATION
=======================

Power Law Detection
-------------------
Many numerical methods exhibit power law convergence behavior of the form:

   error(m) = C × m^(-α)

where:
- m is a parameter (polynomial degree, grid size, iteration count, etc.)
- α is the convergence rate (higher values indicate faster convergence)
- C is a problem-dependent constant

Common convergence rates in numerical analysis:
- O(m^(-1)): Linear methods, basic finite differences
- O(m^(-2)): Quadratic methods, some finite element methods
- O(m^(-4)): Fourth-order methods, high-quality approximations
- O(m^(-8+)): Spectral methods, exceptional approximations

Log-Log Regression Theory
-------------------------
To identify power law relationships, we transform the equation using logarithms:

   error = C × m^(-α)
   log(error) = log(C) + (-α) × log(m)
   log(error) = log(C) - α × log(m)

This transforms the power law into a linear relationship in log-space:
   y = b + mx

where:
- y = log(error)           (dependent variable)
- x = log(m)              (independent variable)
- m = -α                  (slope = negative convergence rate)
- b = log(C)              (intercept = log of constant)

Linear regression in log-space then yields:
- slope = -α  →  convergence rate α = -slope
- intercept = log(C)  →  constant C = exp(intercept)

Quality Assessment
------------------
The quality of a power law fit is assessed using:

1. **R² (coefficient of determination)**:
  - Measures how much variance is explained by the linear model
  - R² ≈ 1.0 indicates excellent power law behavior
  - R² < 0.9 suggests the data may not follow a simple power law

2. **Residual analysis**:
  - Systematic deviations from the fitted line indicate model inadequacy
  - Random scatter around the line confirms power law behavior

3. **Data range**:
  - Power laws should hold across multiple orders of magnitude
  - Fits over narrow ranges may be misleading

IMPLEMENTATION NOTES
====================
This module uses numpy.polynomial.Polynomial for robust linear fitting in log-space.
The choice of error threshold (1e-15) prevents log(0) while being small enough to
capture meaningful convergence behavior above machine precision.
"""
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_errors(biroot, m_values, x_value):
    errors = []
    for m in m_values:
        biroot.change_params(m=m)
        approx = float(biroot(x_value))
        true_val = x_value ** (1 / biroot.n_)
        errors.append(abs(approx - true_val))
    return np.array(errors)


def log_log_fit(m_values, errors, min_points=5):
    # Filter out numerical artifacts and ensure we have valid data
    mask = (errors > 1e-15) & (errors < 1.0) & np.isfinite(errors)
    clean_m = m_values[mask]
    clean_errors = errors[mask]

    # Need sufficient points for reliable fit
    if len(clean_m) < min_points:
        return np.nan

    try:
        # Linear fit in log space: log(error) = slope * log(m) + intercept
        # If error ~ m^(-α), then log(error) ~ -α * log(m) + const
        p = Polynomial.fit(np.log(clean_m), np.log(clean_errors), 1)
        slope = p.coef[1]  # This is -α

        # Calculate R² for quality assessment
        log_m = np.log(clean_m)
        log_errors = np.log(clean_errors)
        predicted = p(log_m)
        ss_res = np.sum((log_errors - predicted) ** 2)
        ss_tot = np.sum((log_errors - np.mean(log_errors)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Only return if we have a good fit
        if r_squared > 0.9:  # Require strong correlation
            return -slope  # Convert to positive convergence rate α
        else:
            return np.nan

    except (np.linalg.LinAlgError, ValueError):
        # Handle numerical issues in fitting
        return np.nan


def analyze_convergence(biroot, m_values, x_values):
    rates = []
    failed_fits = 0

    for x in tqdm(x_values):
        errors = get_errors(biroot, m_values, x)
        rate = log_log_fit(m_values, errors)

        if np.isfinite(rate) and rate > 0:  # Only keep valid positive rates
            rates.append(rate)
        else:
            failed_fits += 1

    if len(rates) == 0:
        return {
            'median': np.nan,
            'min': np.nan,
            'max': np.nan,
            'count': 0,
            'failed_fits': failed_fits,
            'success_rate': 0.0
        }
    return {
        'median': np.median(rates),
        'min': np.min(rates),
        'max': np.max(rates),
        'count': len(rates),
        'failed_fits': failed_fits,
        'success_rate': len(rates) / (len(rates) + failed_fits),
        'rates': rates  # Include raw data for further analysis
    }


def plot_example_fit(biroot, m_vals, x_value=8):
    """Helper function to visualize the power law fit for a specific x value"""
    errors = get_errors(biroot, m_vals, x_value)

    # Filter valid data
    mask = (errors > 1e-15) & (errors < 1.0) & np.isfinite(errors)
    clean_m = m_vals[mask]
    clean_errors = errors[mask]

    if len(clean_m) > 5:
        # Fit and plot
        p = Polynomial.fit(np.log(clean_m), np.log(clean_errors), 1)
        slope = p.coef[1]
        rate = -slope

        plt.figure(figsize=(8, 6))
        plt.loglog(clean_m, clean_errors, 'o-', label=f'Data (x={x_value})')

        # Plot fitted line
        fit_errors = np.exp(p(np.log(clean_m)))
        plt.loglog(clean_m, fit_errors, '--',
                   label=f'Fit: O(m^(-{rate:.1f}))')

        plt.xlabel('m (polynomial degree)')
        plt.ylabel('Absolute error')
        plt.title(f'Power law convergence for x={x_value}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        return rate
    else:
        print(f"Insufficient valid data points for x={x_value}")
        return np.nan


if __name__ == '__main__':
    from biroot import Biroot

    # Test for fourth root (n=4)
    biroot = Biroot(m=20, n=4, c=1)
    m_vals = np.array(range(9, 80))
    x_vals = [0.1, 0.25, 0.75] + [2 ** i for i in range(12)]

    # Run analysis
    stats = analyze_convergence(biroot, m_vals, x_vals)

    print(f"\nResults:")
    print(f"Successful fits: {stats['count']}/{stats['count'] + stats['failed_fits']} ({stats['success_rate']:.1%})")

    if stats['count'] > 0:
        print(f"Convergence rates: O(m^(-α)) where α ∈ [{stats['min']:.1f}, {stats['max']:.1f}]")
        print(f"Median rate: O(m^(-{stats['median']:.1f}))")

        # # Optional: show example fit
        # print(f"\nExample fit for x=8:")
        # example_rate = plot_example_fit(biroot, m_vals, x_value=8)
        # if np.isfinite(example_rate):
        #     print(f"Convergence rate for x=8: O(m^(-{example_rate:.1f}))")
    else:
        print("No valid power-law fits found. Consider:")
        print("- Adjusting the m_values range")
        print("- Checking for numerical issues in biroot evaluation")
        print("- Lowering the R² threshold in log_log_fit()")
