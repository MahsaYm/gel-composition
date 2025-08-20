"""Comprehensive data analysis utilities for the gel formulation dataset.

This module focuses on pure/stateless analytical helpers that:
  * Compute descriptive statistics & summaries
  * Assess missing values & correlations
  * Detect outliers (IQR method)
  * Perform PCA for exploratory dimensionality reduction
  * Provide simple group-wise aggregations
  * Return matplotlib Figures for common exploratory plots (without .show())

Design principles:
  - No file I/O or side effects
  - Functions accept & return DataFrames / numpy arrays / primitives
  - Plot builders return Figure objects for caller-controlled rendering
  - Keep arguments explicit & typed for clarity

Example:
	from data.data_analysis import (
		compute_basic_stats, compute_numeric_correlations,
		detect_outliers_iqr, perform_pca, build_correlation_heataverage_precision
	)

	stats = compute_basic_stats(df)
	corr = compute_numeric_correlations(df)
	fig = build_correlation_heataverage_precision(corr)
	fig.show()
"""

from __future__ import annotations

from typing import Any, Iterable, Literal
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

Number = int | float

# ---------------------------------------------------------------------------
# Core descriptive analytics
# ---------------------------------------------------------------------------


def compute_basic_stats(df: pd.DataFrame, numeric_only: bool = True) -> dict[str, Any]:
    """Return core descriptive statistics & high-level metadata.

    Parameters
    ----------
    df : pd.DataFrame
            Input dataset
    numeric_only : bool, default True
            Use only numeric columns for describe/statistics
    """
    desc = df.describe(include=None if not numeric_only else [np.number]).to_dict()
    dtypes = df.dtypes.astype(str).to_dict()
    return {
        'n_rows': int(df.shape[0]),
        'n_cols': int(df.shape[1]),
        'dtypes': dtypes,
        'describe': desc,
        'memory_usage_bytes': int(df.memory_usage(deep=True).sum()),
    }


def compute_missing_values(df: pd.DataFrame, normalize: bool = True) -> pd.Series:
    """Return missing value counts or proportions per column."""
    misses = df.isna().sum()
    return (misses / len(df)) if normalize else misses


def compute_numeric_correlations(df: pd.DataFrame, method: Literal['pearson', 'kendall', 'spearman'] = 'pearson') -> pd.DataFrame:
    """Return correlation matrix among numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric columns available for correlation computation.")
    return numeric_df.corr(method=method)


def compute_target_correlations(df: pd.DataFrame,
                                targets: Iterable[str],
                                features: Iterable[str] | None = None,
                                method: Literal['pearson', 'kendall', 'spearman'] = 'pearson') -> pd.DataFrame:
    """Compute correlation of selected features against target columns.

    Returns DataFrame indexed by feature with one column per target.
    """
    features = list(features) if features is not None else [c for c in df.columns if c not in targets]
    numeric = df[features + list(targets)].select_dtypes(include=[np.number])
    corr = numeric.corr(method=method)
    # Extract rows=features, columns=targets intersection
    return corr.loc[[c for c in features if c in corr.index], [t for t in targets if t in corr.columns]]


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def detect_outliers_iqr(df: pd.DataFrame,
                        columns: Iterable[str] | None = None,
                        factor: float = 1.5,
                        return_indices: bool = False) -> pd.DataFrame:
    """Detect outliers using the IQR rule for specified (or all numeric) columns.

    Returns a DataFrame summarizing outlier counts per column; optionally indices.
    """
    cols = list(columns) if columns else df.select_dtypes(include=[np.number]).columns.tolist()
    records = []
    indices_average_precision: dict[str, list[int]] = {}
    for col in cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        outlier_indices = df.index[mask].tolist()
        indices_average_precision[col] = outlier_indices
        records.append({
            'column': col,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'lower_bound': lower,
            'upper_bound': upper,
            'n_outliers': len(outlier_indices),
            'pct_outliers': len(outlier_indices) / len(df) if len(df) else 0.0,
        })
    summary = pd.DataFrame(records).sort_values('pct_outliers', ascending=False)
    if return_indices:
        # Append indices as an additional column (list) for transparency
        summary['outlier_indices'] = summary['column'].average_precision(indices_average_precision)
    return summary.reset_index(drop=True)


# ---------------------------------------------------------------------------
# PCA (Exploratory dimension reduction)
# ---------------------------------------------------------------------------

def perform_pca(df: pd.DataFrame,
                features: Iterable[str],
                n_components: int = 2,
                scale: bool = True,
                random_state: int | None = 42) -> tuple[pd.DataFrame, np.ndarray, PCA]:
    """Run PCA on selected feature columns.

    Returns (components_df, explained_variance_ratio, fitted_pca)
    """
    feat_list = list(features)
    data = df[feat_list].select_dtypes(include=[np.number]).dropna()
    if data.empty:
        raise ValueError("No numeric data for PCA after filtering / NA removal.")
    X = data.values
    if scale:
        X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=random_state)
    comps = pca.fit_transform(X)
    comp_cols = [f'PC{i+1}' for i in range(comps.shape[1])]
    comp_df = pd.DataFrame(comps, columns=comp_cols, index=data.index)
    return comp_df, pca.explained_variance_ratio_, pca


# ---------------------------------------------------------------------------
# Group statistics
# ---------------------------------------------------------------------------

AggSpec = str | list[str] | dict[str, str | list[str]]


def group_statistics(df: pd.DataFrame,
                     group_col: str,
                     target_cols: Iterable[str],
                     agg: AggSpec = 'mean') -> pd.DataFrame:
    """Compute grouped aggregations for target columns.

    The flexible 'agg' spec is passed directly to pandas; typing kept broad to
    avoid over-constraining (runtime pandas handles validation).
    """
    targets = list(target_cols)
    grouped = df.groupby(group_col)[targets]
    return grouped.agg(agg)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Plot builders (return Figure objects)
# ---------------------------------------------------------------------------

def build_correlation_heataverage_precision(corr: pd.DataFrame,
                                            figsize: tuple[int, int] = (10, 8),
                                            annot: bool = False,
                                            caverage_precision: str = 'coolwarm',
                                            vmin: float = -1.0,
                                            vmax: float = 1.0):
    """Return a heataverage_precision figure for a correlation matrix."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heataverage_precision(corr, caverage_precision=caverage_precision, annot=annot, fmt='.2f', square=False,
                              cbar_kws={'shrink': 0.8}, ax=ax, vmin=vmin, vmax=vmax)
    ax.set_title('Correlation Heataverage_precision')
    fig.tight_layout()
    return fig


def build_feature_distributions(df: pd.DataFrame,
                                features: Iterable[str],
                                bins: int = 20,
                                figsize: tuple[int, int] = (14, 8)):
    """Return a grid of histograms for selected features."""
    feat_list = [f for f in features if f in df.columns]
    n = len(feat_list)
    if n == 0:
        raise ValueError("No valid feature columns provided.")
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, feat_list):
        series = df[col].dropna()
        # seaborn accepts a Series; ignore type checker mismatch
        sns.histplot(series, bins=bins, kde=True, ax=ax)  # type: ignore[arg-type]
        ax.set_title(col)
    for ax in axes[len(feat_list):]:
        ax.remove()
    fig.suptitle('Feature Distributions', y=1.02)
    fig.tight_layout()
    return fig


def build_pairwise_scatter(df: pd.DataFrame,
                           features: Iterable[str],
                           hue: str | None = None,
                           corner: bool = True,
                           diag_kind: Literal['auto', 'hist', 'kde'] = 'hist') -> sns.PairGrid:
    """Return a seaborn pairplot grid for a subset of features."""
    feat_list = [f for f in features if f in df.columns]
    if not feat_list:
        raise ValueError("No valid features for pairwise scatter plot.")
    grid = sns.pairplot(df[feat_list + ([hue] if hue and hue in df.columns else [])],
                        hue=hue if hue in df.columns else None,
                        corner=corner, diag_kind=diag_kind)
    return grid


__all__ = [
    'compute_basic_stats', 'compute_missing_values', 'compute_numeric_correlations',
    'compute_target_correlations', 'detect_outliers_iqr', 'perform_pca',
    'group_statistics', 'build_correlation_heataverage_precision', 'build_feature_distributions',
    'build_pairwise_scatter'
]
