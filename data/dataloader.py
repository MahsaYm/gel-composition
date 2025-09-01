"""Data loading & preprocessing utilities for the 3D printing gel formulation dataset.

Refactored to expose:
  1. Pure functions (stateless helpers) for each transformation step
  2. A coordinating class `GelDataLoader` wrapping the pipeline

Typical usage:
    loader = GelDataLoader(data_path)
    X, y, meta = loader.load(dataset_types=["Optimization"],
                             input_mode="gel_and_printing",
                             include_leakage=False,
                             target_column=None)
    # After train/test split:
    Xtr_s, Xte_s = loader.scale_inputs(X_train, X_test)
    ytr_s, yte_s = loader.scale_targets(y_train, y_test)
    y_pred_orig = loader.inverse_targets(y_pred_scaled, columns=loader.target_columns)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# input category definitions (constants)
# ---------------------------------------------------------------------------
GEL_FORMULATION_INPUTS: list[str] = [
    'Methocel A4C', 'Methocel A4M', 'Ac-Di-Sol', 'Emulsion'
]
PRINTING_CONDITION_INPUTS: list[str] = [
    'Pressure', 'Speed'
]
PROPERTY_OUTPUTS: list[str] = [
    'Uniformity', 'Disintegration', 'Tanδ', 'Recovery', 'Leakage'
]
METADATA_INPUTS: list[str] = [
    'Name', 'Type', 'Wet tablet'
]

# ---------------------------------------------------------------------------
# Low-level pure helper functions
# ---------------------------------------------------------------------------


def load_raw_data(data_path: str,
                  drop_metadata: bool = False) -> pd.DataFrame:
    """Read CSV into a DataFrame."""
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Data file not found at '{data_path}'") from e
    except Exception as e:  # pragma: no cover (unexpected IO errors)
        raise RuntimeError(f"Error loading data: {e}") from e

    if drop_metadata:
        df = df.drop(columns=METADATA_INPUTS, errors='ignore')

    return df


def filter_by_dataset_types(df: pd.DataFrame,
                            dataset_types: list[str],
                            column_names: list[str]
                            ) -> pd.DataFrame:
    """Return only rows whose Type is in the specified list."""
    filtered = df[df['Type'].isin(dataset_types)].copy()
    return filtered[column_names] if column_names else filtered


def drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows containing any NA values."""
    return df.dropna().copy()


def split_inputs_targets(df: pd.DataFrame,
                         input_columns: list[str],
                         target_column: str
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (X, y)"""
    X = df[input_columns].copy()
    y = df[[target_column]].copy()
    return X, y


def summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Produce simple descriptive summary of a DataFrame."""
    return {
        'total_samples': int(len(df)),
        'dataset_types': df['Type'].value_counts().to_dict() if 'Type' in df.columns else {},
        'missing_values': df.isnull().sum().to_dict(),
        'input_statistics': df.describe().to_dict(),
    }


def scale_train_test(train: pd.DataFrame, test: pd.DataFrame, scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray]:
    """Fit (if needed) a scaler on train and transform both train & test."""
    train_s = scaler.fit_transform(train)
    test_s = scaler.transform(test)
    return train_s, test_s


def inverse_scale(values: np.ndarray, scaler: StandardScaler, columns: list[str]) -> pd.DataFrame:
    """Inverse transform numpy array back to DataFrame with provided column names."""
    inv = scaler.inverse_transform(values)
    return pd.DataFrame(inv, columns=columns)


# ---------------------------------------------------------------------------
# High-level coordinating class
# ---------------------------------------------------------------------------

@dataclass
class GelDataLoader:
    """High-level orchestrator for dataset extraction & scaling.

    Parameters
    ----------
    data_path : str
        Path to CSV dataset.
    cache_raw : bool, default True
        Whether to keep a cached copy of the raw CSV to avoid re-reading.
    """

    data_path: str
    cache_raw: bool = True
    _raw_df: pd.DataFrame | None = field(default=None, init=False, repr=False)
    input_columns: list[str] = field(default_factory=list, init=False)
    target_column: str = field(default=..., init=False)  # type: ignore
    input_scaler: StandardScaler = field(default=StandardScaler(), init=False, repr=False)
    target_scaler: StandardScaler = field(default=StandardScaler(), init=False, repr=False)

    # -------------------------- Loading & filtering ----------------------- #
    def raw(self) -> pd.DataFrame:
        if self._raw_df is None or not self.cache_raw:
            self._raw_df = load_raw_data(self.data_path)
        return self._raw_df

    def load(self,
             dataset_types: list[str],
             input_columns: list[str],
             target_column: str,
             drop_na_rows: bool = True,
             ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        """Main data assembly pipeline.

        Returns
        -------
        X : pd.DataFrame
        y : pd.DataFrame
        meta : dict with keys (input_names, target_names, summary, dataset_types, input_mode)
        """

        self.dataset_types = dataset_types
        self.input_columns = input_columns
        self.target_column = target_column

        df = self.raw()
        df = filter_by_dataset_types(df, dataset_types, input_columns + [target_column])
        if drop_na_rows:
            df = drop_missing(df)

        X, y = split_inputs_targets(df, input_columns, target_column)

        meta = {
            'input_names': input_columns,
            'target_name': target_column,
            'summary': summarize_dataframe(df),
            'dataset_types': dataset_types,
            'shape': {'X': X.shape, 'y': y.shape},
        }
        return X, y, meta

    # ----------------------------- Scaling -------------------------------- #
    def scale_inputs(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        train_s, test_s = scale_train_test(X_train, X_test, self.input_scaler)
        return train_s, test_s

    def scale_targets(self, y_train: pd.DataFrame, y_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        train_s, test_s = scale_train_test(y_train, y_test, self.target_scaler)
        return train_s, test_s

    # ---------------------------- Inference ------------------------------- #
    def inverse_targets(self, values: np.ndarray, target_column: str) -> pd.DataFrame:
        if self.target_scaler is None:
            raise RuntimeError("Target scaler not fit yet. Call scale_targets first.")
        return inverse_scale(values, self.target_scaler, [target_column])

    # ----------------------------- Utilities ------------------------------ #
    def get_summary(self) -> dict[str, Any]:
        return summarize_dataframe(self.raw())

    def get_signature(self) -> str:
        return f'{self.dataset_types=}, {self.input_columns=}, {self.target_column=}'


__all__ = [
    'GelDataLoader',
    'load_raw_data', 'filter_by_dataset_types', 'drop_missing',
    'split_inputs_targets', 'scale_train_test'
]
