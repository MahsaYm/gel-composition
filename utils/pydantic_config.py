"""Pydantic configuration utilities for handling NumPy arrays and complex types.

This module provides reusable configurations and validators for Pydantic models
that need to work with NumPy arrays and other scientific computing types.
"""

from typing import Any
import numpy as np
from pydantic import ConfigDict, field_validator


def numpy_compatible_config() -> ConfigDict:
    """Return a Pydantic ConfigDict that allows NumPy arrays and other arbitrary types.
    
    Use this configuration for any Pydantic model that needs to store:
    - NumPy arrays
    - scikit-learn objects
    - Other complex scientific computing objects
    
    Example:
        class MyModel(BaseModel):
            model_config = numpy_compatible_config()
            
            data: np.ndarray
            predictions: np.ndarray
    """
    return ConfigDict(
        arbitrary_types_allowed=True,
        # Allow extra fields for flexibility
        extra='forbid',
        # Validate assignment to catch errors early
        validate_assignment=True,
    )


class NumpyArrayValidator:
    """Validators for NumPy arrays in Pydantic models."""
    
    @staticmethod
    @field_validator('*', mode='before')
    def validate_numpy_array(v: Any) -> Any:
        """Convert lists to NumPy arrays and validate shape."""
        if isinstance(v, (list, tuple)):
            return np.asarray(v)
        return v
    
    @staticmethod
    def validate_shape(expected_shape: tuple[int, ...]):
        """Create a validator that checks array shape."""
        def validator(v: np.ndarray) -> np.ndarray:
            if hasattr(v, 'shape') and v.shape != expected_shape:
                raise ValueError(f"Expected shape {expected_shape}, got {v.shape}")
            return v
        return field_validator('*', mode='after')(validator)


# Example usage
if __name__ == "__main__":
    from pydantic import BaseModel
    
    class ExampleModel(BaseModel):
        model_config = numpy_compatible_config()
        
        predictions: np.ndarray
        targets: np.ndarray
        metadata: dict[str, Any]
    
    # Test it works
    model = ExampleModel(
        predictions=np.array([1.0, 2.0, 3.0]),
        targets=np.array([1.1, 2.1, 3.1]),
        metadata={"experiment": "test"}
    )
    
    print("✅ NumPy arrays work perfectly!")
    print(f"Predictions: {model.predictions}")
    print(f"Targets: {model.targets}")
