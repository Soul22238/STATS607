import pytest
import numpy as np
from bootstrap import bootstrap_sample, bootstrap_ci, r_squared

class TestBootstrap:
    """Test suite for bootstrap functions"""
    
    def test_bootstrap_sample_basic(self):
        """Test basic functionality of bootstrap_sample"""
        np.random.seed(42)  # For reproducibility
        X = np.column_stack((np.ones(10), np.arange(10)))  # Intercept + linear trend
        y = 2 + 3 * np.arange(10) + np.random.normal(0, 1, size=10)  # Known relationship
        
        def compute_stat(X, y):
            return r_squared(X, y)
        
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
            return r_squared(X, y)
        
        # Zero variance in y
        with pytest.raises(ValueError, match="y has zero variance"):
            bootstrap_sample(X, y, compute_stat, n_bootstrap=1000)

        # Array length mismatch
        X_mismatch = np.column_stack((np.ones(4), np.arange(4)))
        y_mismatch = np.array([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="Number of rows in X must equal length of y"):
            bootstrap_sample(X_mismatch, y_mismatch, compute_stat, n_bootstrap=1000)
        
        # Empty arrays
        X_empty = np.empty((0, 2))
        y_empty = np.array([])
        with pytest.raises(ValueError, match="Empty arrays"):
            bootstrap_sample(X_empty, y_empty, compute_stat, n_bootstrap=1000)
    
    def test_bootstrap_sample_warning(self):
        """Test that warnings are issued appropriately."""
        def compute_stat(X, y):
            return r_squared(X, y)
        # Test warning about small sample size
        X_small = np.column_stack((np.ones(5), np.arange(5)))
        y_small = np.array([1, 2, 3, 4, 5])
        with pytest.warns(UserWarning, match="Small sample size"):
            bootstrap_sample(X_small, y_small, compute_stat, n_bootstrap=1000)
        
        # Test warning about few bootstrap samples
        with pytest.warns(UserWarning, match="Number of bootstrap samples is low"):
            bootstrap_ci([1, 2, 3, 4, 5] * 10, np.mean, n_bootstrap=100)
        
    


def test_bootstrap_integration():
    """Test that bootstrap_sample and bootstrap_ci work together"""
    # This test should initially fail
    pass

# TODO: Add your unit tests here

