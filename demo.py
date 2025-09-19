import numpy as np
import warnings
import statsmodels.api as sm

def calculate_correlation(x, y):
    """Calculate Pearson correlation coefficient with robust error handling."""
    # Input validation with specific error types
    if not isinstance(x, (list, np.ndarray)) or not isinstance(y, (list, np.ndarray)):
        raise TypeError("Both x and y must be arrays or lists")
    
    x, y = np.asarray(x), np.asarray(y)
    
    if len(x) != len(y):
        raise ValueError(f"Arrays must have same length: got {len(x)} and {len(y)}")
    
    if len(x) < 2:
        raise ValueError("Need at least 2 observations to compute correlation")
    
    # Check for numerical issues
    if np.var(x) == 0:
        raise ValueError("Cannot compute correlation: x has zero variance")
    if np.var(y) == 0:
        raise ValueError("Cannot compute correlation: y has zero variance")
    
    # Check for missing values - more informative than letting numpy fail
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        raise ValueError("Cannot compute correlation with missing values (NaN)")
    
    return np.corrcoef(x, y)[0, 1]

def fit_linear_model(x, y):
    """Simple linear regression using statsmodels."""
    x_with_const = sm.add_constant(x)
    model = sm.OLS(y, x_with_const)
    return model.fit()

def robust_regression_analysis(data, group_col, outcome_col, predictor_col):
    """Fit regression models by group with proper error handling."""
    results = {}
    
    for group in data[group_col].unique():
        subset = data[data[group_col] == group]
        
        try:
            # Check minimum sample size
            if len(subset) < 3:
                warnings.warn(f"Group {group} has only {len(subset)} observations. "
                            f"Results may be unreliable.", UserWarning)
            
            # Fit model with error handling
            model = fit_linear_model(subset[predictor_col], subset[outcome_col])
            results[group] = model
            
        except Exception as e:
            # Re-raise with more context (following Paciorek's pattern)
            print(f"Regression failed for group {group}. "
                  f"Check data quality for this subset.")
            raise  # Re-raise the original exception
    
    return results

def bootstrap_ci(data, statistic_func, n_bootstrap=1000, alpha=0.05):
    """Calculate bootstrap confidence interval with appropriate warnings."""
    
    # Warn about potential issues
    if len(data) < 30:
        warnings.warn(f"Small sample size (n={len(data)}). "
                     f"Bootstrap may be unreliable.", UserWarning)
    
    if n_bootstrap < 1000:
        warnings.warn(f"Using {n_bootstrap} bootstrap samples. "
                     f"Consider using at least 1000.", UserWarning)
    
    # Check for numerical stability issues - fix division by zero
    data_mean = np.mean(data)
    if abs(data_mean) > 1e-10:  # Avoid division by very small numbers
        coeff_var = np.std(data) / abs(data_mean)
        if coeff_var > 2.0:
            warnings.warn("Data has high variability. Bootstrap CI may be wide.", 
                         UserWarning)
    
    # Perform bootstrap
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic_func(sample))
    
    # Calculate CI
    lower = np.percentile(bootstrap_stats, 100 * alpha/2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha/2))
    
    return lower, upper