"""Custom scalers for machine learning pipelines."""
from __future__ import annotations

import numpy as np


class IdentityScaler:
    """A no-op scaler that returns data unchanged.
    
    Useful for classification tasks where target scaling is not needed,
    while maintaining a consistent interface with StandardScaler.
    """
    
    def fit(self, X: np.ndarray) -> IdentityScaler:
        """Fit the scaler (no-op for identity scaling)."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data (returns unchanged)."""
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform the data (returns unchanged)."""
        return X
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform the data (returns unchanged)."""
        return X
