from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


def _to_query_series(query_instance: Union[pd.DataFrame, pd.Series]) -> pd.Series:
    if isinstance(query_instance, pd.DataFrame):
        if query_instance.empty:
            raise ValueError("query_instance dataframe is empty.")
        return query_instance.iloc[0]
    if isinstance(query_instance, pd.Series):
        return query_instance
    raise TypeError("query_instance must be a pandas Series or single-row DataFrame.")


def _get_numeric_feature_subset(
    query_series: pd.Series,
    counterfactuals: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
) -> Sequence[str]:
    if features is not None:
        selected = [
            col for col in features if col in counterfactuals.columns and col in query_series.index
        ]
    else:
        selected = [
            col for col in counterfactuals.columns if col in query_series.index
        ]

    numeric_selected = []
    for col in selected:
        if pd.api.types.is_numeric_dtype(counterfactuals[col]):
            numeric_selected.append(col)
    if not numeric_selected:
        raise ValueError("No numeric features available for plotting.")
    return numeric_selected


def plot_counterfactual_deltas(
    query_instance: Union[pd.DataFrame, pd.Series],
    counterfactuals: pd.DataFrame,
    top_k: int = 20,
    figsize: Tuple[float, float] = (12, 5),
    ax: Optional[plt.Axes] = None,
):
    """
    Plot mean feature deltas (CF - query) for the most-changed features.
    """
    query_series = _to_query_series(query_instance)
    features = _get_numeric_feature_subset(query_series=query_series, counterfactuals=counterfactuals)

    cf_values = counterfactuals[features].astype(float)
    query_values = query_series[features].astype(float)
    deltas = cf_values.subtract(query_values, axis=1)
    top_features = deltas.abs().mean().sort_values(ascending=False).head(top_k).index.tolist()
    mean_delta = deltas[top_features].mean().sort_values()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    colors = np.where(mean_delta.values >= 0, "#2E8B57", "#B22222")
    ax.barh(mean_delta.index, mean_delta.values, color=colors)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_title("Average Counterfactual Feature Delta")
    ax.set_xlabel("Mean Delta (CF - Query)")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    return fig, ax


def plot_counterfactual_profiles(
    query_instance: Union[pd.DataFrame, pd.Series],
    counterfactuals: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    top_k: int = 20,
    max_counterfactuals: int = 6,
    figsize: Tuple[float, float] = (14, 5),
    ax: Optional[plt.Axes] = None,
):
    """
    Plot query and counterfactual profiles over selected features.
    """
    query_series = _to_query_series(query_instance)
    numeric_features = _get_numeric_feature_subset(
        query_series=query_series,
        counterfactuals=counterfactuals,
        features=features,
    )

    if features is None:
        deltas = counterfactuals[numeric_features].astype(float).subtract(
            query_series[numeric_features].astype(float),
            axis=1,
        )
        numeric_features = (
            deltas.abs().mean().sort_values(ascending=False).head(top_k).index.tolist()
        )
    else:
        numeric_features = list(numeric_features)[:top_k]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = np.arange(len(numeric_features))
    query_values = query_series[numeric_features].astype(float).values
    ax.plot(x, query_values, color="black", linewidth=2.2, label="Query")

    cf_subset = counterfactuals[numeric_features].head(max_counterfactuals)
    for idx, (_, row) in enumerate(cf_subset.iterrows(), start=1):
        ax.plot(x, row.astype(float).values, linewidth=1.2, alpha=0.7, label=f"CF {idx}")

    ax.set_xticks(x)
    ax.set_xticklabels(numeric_features, rotation=45, ha="right")
    ax.set_title("Query vs Counterfactual Profiles")
    ax.set_ylabel("Feature Value")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_time_series_counterfactual_grid(
    counterfactuals: pd.DataFrame,
    series_builder: Callable[[pd.Series, str], Tuple[pd.DataFrame, pd.Series, pd.Series, float, float]],
    flux_types: Sequence[str] = ("p3_flux_ic", "p5_flux_ic", "p7_flux_ic"),
    max_counterfactuals: int = 10,
    ncols: int = 2,
    window_size: int = 10,
    figsize: Tuple[float, float] = (14, 10),
    log_scale: bool = True,
    y_min: float = 0.01,
):
    """
    Plot grid of original vs counterfactual time-series traces.

    `series_builder` should return:
    (time_dataframe, cf_series, original_series, min_y, max_y)
    for a given (counterfactual_row, flux_type).
    """
    if counterfactuals.empty:
        raise ValueError("counterfactuals dataframe is empty.")
    if not flux_types:
        raise ValueError("flux_types must contain at least one series name.")

    nplots = min(max_counterfactuals, len(counterfactuals))
    nrows = int(math.ceil(nplots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, facecolor="white")
    axes_flat = axes.flatten()

    colors = {
        "p3_flux_ic": "#FF0000",
        "p5_flux_ic": "#1b5c0c",
        "p7_flux_ic": "#FFA500",
    }

    for cf_idx in range(nplots):
        sample = counterfactuals.iloc[cf_idx]
        ax = axes_flat[cf_idx]
        global_max_y = y_min

        for flux_type in flux_types:
            df_obs, cf_series, original_series, _, max_y = series_builder(sample, flux_type)
            original_smooth = pd.Series(original_series).rolling(
                window=window_size, center=True
            ).mean()
            cf_smooth = pd.Series(cf_series).rolling(window=window_size, center=True).mean()

            color = colors.get(flux_type, None)
            ax.plot(
                df_obs["time_stamp"],
                original_smooth,
                color=color,
                label=f"Original {flux_type}",
            )
            ax.plot(
                df_obs["time_stamp"],
                cf_smooth,
                color=color,
                linestyle="dashed",
                label=f"Counterfactual {flux_type}",
            )
            global_max_y = max(global_max_y, float(max_y))

        if log_scale:
            ax.set_yscale("log")
        ax.set_ylim(bottom=y_min, top=max(global_max_y, y_min * 10))
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

        if "time_stamp" in df_obs.columns:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_title(f"CF #{cf_idx + 1}")

    for idx in range(nplots, len(axes_flat)):
        axes_flat[idx].axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), frameon=False)
    fig.supxlabel("Period of Observation")
    fig.supylabel("Particle Flux Unit")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig, axes
