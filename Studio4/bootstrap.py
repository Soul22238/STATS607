import numpy as np

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
    # Create container for the boostraping statistics
    boost_stat = np.zeros(n_bootstrap)
    
    n = len(y)
    
    # Loop to calculate statistic
    for i in range(n_bootstrap):
        idx = np.random.choice(n)
        boost_stat[i] = compute_stat(X[idx, ], y[idx])
    
    return boost_stat

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
    if len(bootstrap_stats) == 0:
        raise ValueError("bootstrap_stats must not be empty")
    
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
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != len(y):
        raise ValueError("Number of rows in X must equal length of y")

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