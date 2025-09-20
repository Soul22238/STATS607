import pytest
import numpy as np

from scipy import stats
from bootstrap import bootstrap_sample, bootstrap_ci, R_squared

class TestBootstrap:
    """Test suite for bootstrap functions"""
    
    def test_bootstrap_sample_basic(self):
        """Test basic functionality of bootstrap_sample."""
        np.random.seed(42)  # For reproducibility
        X = np.column_stack((np.ones(10), np.arange(10)))  # Intercept + linear trend
        y = 2 + 3 * np.arange(10) + np.random.normal(0, 1, size=10)  # Known relationship
        
        def compute_stat(X, y):
            return R_squared(X, y)
        
        # Original R-squared value
        original_r2 = compute_stat(X, y)
        
        # Generate bootstrap samples
        stats = bootstrap_sample(X, y, compute_stat, n_bootstrap=1001)
        
        # Basic checks
        assert len(stats) == 1001
        assert np.all(np.isfinite(stats))
        
        # Statistical properties checks
        assert 0 <= np.mean(stats) <= 1  # R-squared should be between 0 and 1
        assert np.abs(np.mean(stats) - original_r2) < 0.2  # Bootstrap mean should be close to original
        assert np.std(stats) > 0  # Should have some variation
        assert len(np.unique(stats)) > 50  # Should have many unique values
    
    def test_bootstrap_sample_edge_cases(self):
        """Test edge cases that commonly cause bugs."""
        X = np.column_stack((np.ones(5), np.arange(5)))
        y = np.array([1, 1, 1, 1, 1])  # Zero variance in y
        
        def compute_stat(X, y):
            return R_squared(X, y)
        
        # Zero variance in y
        with pytest.raises(ValueError, match=r"y has zero variance"):
            bootstrap_sample(X, y, compute_stat, n_bootstrap=1000)

        # Array length mismatch
        X_mismatch = np.column_stack((np.ones(4), np.arange(4)))
        y_mismatch = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match=f"X and y must have same length: got 4 and 5"):
            bootstrap_sample(X_mismatch, y_mismatch, compute_stat, n_bootstrap=1000)
        
        # Empty arrays
        X_empty = np.empty((0, 2))
        y_empty = np.array([])
        with pytest.raises(ValueError, match="X and y must not be empty"):
            bootstrap_sample(X_empty, y_empty, compute_stat, n_bootstrap=1000)
    
    def test_bootstrap_sample_warning(self):
        """Test that warnings are issued appropriately."""
        def compute_stat(X, y):
            return R_squared(X, y)
        # Test warning about small sample size
        X_small = np.column_stack((np.ones(5), np.arange(5)))
        y_small = np.array([1, 2, 3, 4, 5])
        with pytest.warns(UserWarning, match="X and y have a small sample size of 5."):
            bootstrap_sample(X_small, y_small, compute_stat, n_bootstrap=1000)
        
        # Test warning about few bootstrap samples
        X = np.column_stack((np.ones(10), np.arange(10)))  # Intercept + linear trend
        y = 2 + 3 * np.arange(10) + np.random.normal(0, 1, size=10)  # Known relationship
        with pytest.warns(UserWarning, match="X and y have a small sample size"):
            bootstrap_sample(X, y, compute_stat, n_bootstrap=100)
    
    def test_bootstrap_ci_happy_path(self):
        """Test basic functionality of bootstrap_ci."""
        stats = np.arange(100)
        ci = bootstrap_ci(stats, alpha=0.1)
        
        assert len(ci) == 2
        assert 0 <= ci[0] <= 10
        assert 90 <= ci[1] <= 100
        assert isinstance(ci, tuple)

    # def test_bootstrap_ci_edge_cases(self):
    #     """Test edge case for quantile functions."""
    #     stats = np.array([1])

    #     ci = bootstrap_ci(stats, alpha=0.05)

    #     assert isinstance(ci, tuple)
    #     assert len(ci) == 2
    #     assert ci[0] == 1
    #     assert ci[1] == 1

    def test_bootstrap_ci_errors(self):
        """Test input validation."""
        with pytest.raises(ValueError, match="bootstrap_stats must not be empty"):
            bootstrap_ci([], alpha=0.05)
        
        with pytest.raises(ValueError, match="alpha must be in \(0, 1\)"):
            bootstrap_ci(np.arange(10), alpha=-0.1)
        
        with pytest.raises(ValueError, match="alpha must be in \(0, 1\)"):
            bootstrap_ci(np.arange(10), alpha=1.0)

    def test_bootstrap_ci_wanings(self):
        """Test insufficient sample case."""
        stats = np.arange(500)
        with pytest.warns(UserWarning, match="Only 500 bootstrap samples; CI may be inaccurate"):
            bootstrap_ci(stats, alpha=0.05)

    def test_r_square_happy_path(self):
        """Test basic functionality of R_square."""
        X = np.array([[1, 1],
                      [1, 2],
                      [1, 3],
                      [1, 4]])  # intercept + one feature
        y = np.array([2, 3, 4, 5])
        assert round(R_squared(X, y), 2) == 1.0

        y_noisy = y + np.array([0.1, -0.1, 0.2, -0.2])
        r2_noisy = R_squared(X, y_noisy)
        assert 0.9 < r2_noisy <= 1.0

    def test_r_square_errors(self):
        """Test input validation."""
        # Mismatched X and y
        X = np.array([[1,1],[1,2]])
        y = np.array([1,2,3])
        with pytest.raises(ValueError, match="Number of rows in X must equal length of y"):
            R_squared(X, y)
        
        # First column not all ones
        X_bad = np.array([[0,1],[1,2]])
        y = np.array([1,2])
        with pytest.raises(ValueError, match="First column of X should be all ones to include intercept"):
            R_squared(X_bad, y)
        
        # Zero variance
        X = np.array([[1,1],[1,2],[1,3]])
        y_const = np.array([5,5,5])
        with pytest.raises(ValueError, match="y has zero variance; R-squared is undefined"):
            R_squared(X, y_const)
        
        # Collinear features (singular X^T X)
        X_collinear = np.array([[1,1,2],
                                [1,2,4],
                                [1,3,6]])
        y = np.array([1,2,3])
        with pytest.raises(np.linalg.LinAlgError, match="Matrix X\^T X is singular; features may be collinear"):
            R_squared(X_collinear, y)


# def test_bootstrap_integration():
#     """Test that bootstrap_sample and bootstrap_ci work together"""
#     # This test should initially fail
#     pass
def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create synthetic data with known relationship
    n = 50  # sample size
    X = np.column_stack([
        np.ones(n),  # intercept
        np.linspace(0, 10, n)  # feature
    ])
    true_beta = np.array([1, 2])  # true coefficients
    y = X @ true_beta + np.random.normal(0, 1, n)  # y = 1 + 2x + noise
    
    # Calculate original R-squared
    original_r2 = R_squared(X, y)
    
    # Generate bootstrap distribution
    n_bootstrap = 1000
    bootstrap_stats = bootstrap_sample(
        X, y,
        compute_stat=R_squared,
        n_bootstrap=n_bootstrap
    )
    
    # Calculate 95% confidence interval
    ci = bootstrap_ci(bootstrap_stats, alpha=0.05)
    
    # Assertions
    assert len(bootstrap_stats) == n_bootstrap
    assert isinstance(ci, tuple)
    assert len(ci) == 2
    assert ci[0] <= original_r2 <= ci[1]  # original stat should be in CI
    assert 0 <= ci[0] <= ci[1] <= 1  # R-squared CI should be in [0,1]
    assert len(np.unique(bootstrap_stats)) > n_bootstrap/10  # Check for variation
    assert np.mean(bootstrap_stats) > 0.5  # Should have good fit

def test_bootstrap_alternative_distribution():
    """
    Test bootstrap R² distribution under the alternative hypothesis
    (when X and y are correlated).
    
    - Under H0 (independence), R² ~ Beta(p/2, (n-p-1)/2).
    - Under H1 (dependence), the distribution of R² deviates from Beta.
    - Therefore, the KS test p-value should be small (< 0.01).
    """
    

    np.random.seed(42)
    
    # Setup parameters
    n = 100   # sample size
    p = 4     # number of predictors (excluding intercept)
    n_bootstrap = 1000
    
    # Generate predictors X (with intercept in the first column)
    X = np.column_stack([
        np.ones(n),                        # intercept column
        np.random.normal(0, 1, (n, p))     # random predictors
    ])
    
    # Define true regression coefficients (including intercept)
    beta = np.array([1.0, 0.5, -0.3, 0.8, 0.0])  
    
    # Generate response y = X @ beta + noise (dependent on X)
    noise = np.random.normal(0, 1, n)
    y = X @ beta + noise
    
    # Bootstrap distribution of R²
    bootstrap_stats = bootstrap_sample(
        X, y,
        compute_stat=R_squared,
        n_bootstrap=n_bootstrap
    )
    
    # Theoretical Beta distribution (valid only under null)
    alpha = p / 2
    beta_param = (n - p - 1) / 2
    theoretical_dist = stats.beta(alpha, beta_param)
    
    # Kolmogorov-Smirnov test
    ks_stat, p_value = stats.kstest(bootstrap_stats, theoretical_dist.cdf)
    
    # Since X and y are correlated, R² does NOT follow Beta
    # Expect rejection of null: p-value should be very small
    assert p_value < 0.01, (
        f"Expected bootstrap distribution to deviate from Beta under alternative, "
        f"but got KS test p-value = {p_value:.4f}"
    )
