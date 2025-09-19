import numpy as np
import warnings
"""
Strong linear model in regression
    Y = X beta + eps, where eps~ N(0, sigma^2 I)
    Under the null where beta_1 = ... = beta_p = 0,
    the R-squared coefficient has a known distribution
    (if you have an intercept beta_0), 
        R^2 ~ Beta(p/2, (n-p-1)/2)
"""


def bootstrap_sample(X, y, compute_stat, n_bootstrap=1000):
    """
    Generate bootstrap distribution of a statistic

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)
    compute_stat : callable
        Function that computes a statistic (float) from data (X, y)
    n_bootstrap : int, default 1000
        Number of bootstrap samples to generate

    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics, length n_bootstrap

    ....
    """
    pass

def bootstrap_ci(bootstrap_stats, alpha=0.05):
    """
    Calculate confidence interval from the bootstrap samples

    Parameters
    ----------
    bootstrap_stats : array-like
        Array of bootstrap statistics
    alpha : float, default 0.05
        Significance level (e.g. 0.05 gives 95% CI)

    Returns
    -------
    tuple 
        (lower_bound, upper_bound) of the CI
    
    Raises
    ------
    ValueError
        If alpha is not in (0, 1)
        If bootstrap_stats is empty
    
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    # Warn about potential issues

    if len(bootstrap_stats) == 0:
        raise ValueError("bootstrap_stats must not be empty")
    elif len(bootstrap_stats) < 1000:
        warnings.warn(f"Only {len(bootstrap_stats)} bootstrap samples; CI may be inaccurate", UserWarning)
    
    lower = np.percentile(bootstrap_stats, 100 * (alpha / 2))
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    return (lower, upper)

def R_squared(X, y):
    """
    Calculate R-squared from multiple linear regression.

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)

    Returns
    -------
    float
        R-squared value (between 0 and 1) from OLS
    
    Raises
    ------
    ValueError
        If X.shape[0] != len(y)
        If first column of X is not all ones (intercept)
        If y has zero variance
    LinAlgError
        If X^T X is singular (features may be collinear)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != len(y):
        raise ValueError("Number of rows in X must equal length of y")
    if not np.allclose(X[:,0], 1):
        raise ValueError("First column of X should be all ones to include intercept")
    if np.linalg.matrix_rank(X.T @ X) < X.shape[1]:
        raise np.linalg.LinAlgError("Matrix X^T X is singular; features may be collinear")
    if np.allclose(y, y.mean()):
        raise ValueError("y has zero variance; R-squared is undefined")


    # OLS estimate of coefficients: beta = (X^T X)^{-1} X^T y
    beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

    # Predictions
    y_hat = X @ beta_hat

    # Total sum of squares
    y_mean = np.mean(y)
    SST = np.sum((y - y_mean) ** 2)

    # Residual sum of squares
    SSR = np.sum((y - y_hat) ** 2)

    # R-squared
    R2 = 1 - SSR / SST
    return R2