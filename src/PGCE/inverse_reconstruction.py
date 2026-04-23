from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InverseReconstructionResult:
    df_obs: pd.DataFrame
    reconstructed: Dict[str, pd.Series]
    original: Dict[str, pd.Series]
    diagnostics: Dict[str, Dict[str, float]]


def _extract_target_mean(sample_cfe: pd.Series, flux_type: str, start: int, end: int) -> Optional[float]:
    key = f"{flux_type}_mean@[{start}:{end}]"
    if key not in sample_cfe.index:
        return None
    value = sample_cfe[key]
    if pd.isna(value):
        return None
    return float(value)


def _build_window_operator(minutes: np.ndarray, slices: Sequence[Tuple[int, int]]) -> Tuple[np.ndarray, List[Tuple[int, int]], List[np.ndarray]]:
    rows = []
    used_slices: List[Tuple[int, int]] = []
    masks: List[np.ndarray] = []

    for start, end in slices:
        mask = (minutes >= start) & (minutes < end)
        count = int(mask.sum())
        if count == 0:
            continue
        row = mask.astype(float) / count
        rows.append(row)
        used_slices.append((int(start), int(end)))
        masks.append(mask)

    if not rows:
        return np.zeros((0, len(minutes)), dtype=float), used_slices, masks
    return np.vstack(rows).astype(float), used_slices, masks


def _second_difference_matrix(n: int) -> np.ndarray:
    if n <= 2:
        return np.zeros((n, n), dtype=float)
    d = np.zeros((n - 2, n), dtype=float)
    for i in range(n - 2):
        d[i, i] = 1.0
        d[i, i + 1] = -2.0
        d[i, i + 2] = 1.0
    return d.T @ d


def _window_mean_mae(
    series: np.ndarray,
    minutes: np.ndarray,
    slices: Sequence[Tuple[int, int]],
    sample_cfe: pd.Series,
    flux_type: str,
) -> Tuple[float, int]:
    errs = []
    used = 0
    for start, end in slices:
        target = _extract_target_mean(sample_cfe, flux_type, start, end)
        if target is None:
            continue
        mask = (minutes >= start) & (minutes < end)
        if not np.any(mask):
            continue
        actual = float(np.mean(series[mask]))
        errs.append(abs(actual - target))
        used += 1
    if used == 0:
        return 0.0, 0
    return float(np.mean(errs)), used


def inverse_reconstruct_counterfactual_series(
    csv_path: str,
    sample_cfe: pd.Series,
    slices: Sequence[Tuple[int, int]],
    flux_types: Sequence[str] = ("p3_flux_ic", "p5_flux_ic", "p7_flux_ic"),
    start_offset_min: int = 300,
    end_offset_min: int = 660,
    delim: str = ",",
    window_weight: float = 15.0,
    proximity_weight: float = 8.0,
    smoothness_weight: float = 2.0,
    ordering_penalty: float = 50.0,
    ordering_margin: float = 0.0,
    p3_upper_bound: Optional[float] = 10.0,
    threshold_penalty: float = 20.0,
    min_value: Optional[float] = 1e-6,
    max_value: Optional[float] = None,
    learning_rate: float = 0.01,
    max_iter: int = 500,
    tol: float = 1e-7,
) -> InverseReconstructionResult:
    """
    Reconstructs full time-series curves from CF window-mean targets by solving
    a global inverse optimization problem with soft constraints.

    Objective combines:
    - window-mean fidelity to CF features
    - proximity to original observed series
    - smoothness regularization
    - optional ordering (p3 > p5 > p7) and p3 upper-bound penalties
    """
    df = pd.read_csv(csv_path, delimiter=delim).rename(columns={"time_tag": "time_stamp"})
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], format="%Y-%m-%d %H:%M:%S")

    event_start = df["time_stamp"].iloc[0] + timedelta(minutes=start_offset_min)
    event_end = df["time_stamp"].iloc[0] + timedelta(minutes=end_offset_min)
    df_obs = df[(df["time_stamp"] >= event_start) & (df["time_stamp"] < event_end)].copy()
    df_obs["minutes"] = (df_obs["time_stamp"] - event_start).dt.total_seconds() / 60.0

    minutes = df_obs["minutes"].to_numpy(dtype=float)
    n_t = len(df_obs)
    n_flux = len(flux_types)
    if n_t == 0:
        raise ValueError("No observations found in selected reconstruction window.")

    # Build linear window operator once (same masks for all channels).
    A, used_slices, _ = _build_window_operator(minutes, slices)
    n_windows = A.shape[0]

    x0 = np.vstack([df_obs[flux].to_numpy(dtype=float) for flux in flux_types])
    x = x0.copy()

    # Targets per channel for slices that exist in both time window and CF vector.
    targets = np.zeros((n_flux, n_windows), dtype=float)
    target_mask = np.zeros((n_flux, n_windows), dtype=bool)
    for c, flux in enumerate(flux_types):
        for w, (start, end) in enumerate(used_slices):
            tval = _extract_target_mean(sample_cfe, flux, start, end)
            if tval is None:
                continue
            targets[c, w] = tval
            target_mask[c, w] = True

    L = _second_difference_matrix(n_t)

    flux_to_idx: Mapping[str, int] = {name: idx for idx, name in enumerate(flux_types)}
    p3_idx = flux_to_idx.get("p3_flux_ic", None)
    p5_idx = flux_to_idx.get("p5_flux_ic", None)
    p7_idx = flux_to_idx.get("p7_flux_ic", None)

    prev_loss = np.inf
    for _ in range(max_iter):
        grad = np.zeros_like(x, dtype=float)
        total_loss = 0.0

        # Per-channel quadratic terms
        for c in range(n_flux):
            if n_windows > 0 and np.any(target_mask[c]):
                pred = A @ x[c]
                diff = np.zeros(n_windows, dtype=float)
                diff[target_mask[c]] = pred[target_mask[c]] - targets[c, target_mask[c]]
                total_loss += window_weight * float(np.dot(diff, diff))
                grad[c] += 2.0 * window_weight * (A.T @ diff)

            prox = x[c] - x0[c]
            total_loss += proximity_weight * float(np.dot(prox, prox))
            grad[c] += 2.0 * proximity_weight * prox

            smooth = L @ x[c]
            total_loss += smoothness_weight * float(np.dot(x[c], smooth))
            grad[c] += 2.0 * smoothness_weight * smooth

        # Coupled ordering constraints: p3 > p5 > p7
        if p3_idx is not None and p5_idx is not None and p7_idx is not None:
            v1 = x[p5_idx] - x[p3_idx] + ordering_margin
            mask1 = v1 > 0
            if np.any(mask1):
                v = v1[mask1]
                total_loss += ordering_penalty * float(np.dot(v, v))
                grad[p3_idx, mask1] += -2.0 * ordering_penalty * v
                grad[p5_idx, mask1] += 2.0 * ordering_penalty * v

            v2 = x[p7_idx] - x[p5_idx] + ordering_margin
            mask2 = v2 > 0
            if np.any(mask2):
                v = v2[mask2]
                total_loss += ordering_penalty * float(np.dot(v, v))
                grad[p5_idx, mask2] += -2.0 * ordering_penalty * v
                grad[p7_idx, mask2] += 2.0 * ordering_penalty * v

        # Optional p3 upper bound
        if p3_idx is not None and p3_upper_bound is not None:
            excess = x[p3_idx] - p3_upper_bound
            mask = excess > 0
            if np.any(mask):
                v = excess[mask]
                total_loss += threshold_penalty * float(np.dot(v, v))
                grad[p3_idx, mask] += 2.0 * threshold_penalty * v

        x = x - learning_rate * grad

        # Projection to physically plausible range.
        if min_value is not None:
            x = np.maximum(x, float(min_value))
        if max_value is not None:
            x = np.minimum(x, float(max_value))

        if abs(prev_loss - total_loss) < tol:
            break
        prev_loss = total_loss

    reconstructed = {
        flux: pd.Series(x[idx], index=df_obs.index)
        for idx, flux in enumerate(flux_types)
    }
    original = {
        flux: pd.Series(x0[idx], index=df_obs.index)
        for idx, flux in enumerate(flux_types)
    }

    diagnostics: Dict[str, Dict[str, float]] = {}
    for idx, flux in enumerate(flux_types):
        mae, used = _window_mean_mae(
            series=x[idx],
            minutes=minutes,
            slices=slices,
            sample_cfe=sample_cfe,
            flux_type=flux,
        )
        diagnostics[flux] = {
            "window_mean_mae": float(mae),
            "used_windows": float(used),
        }

    if p3_idx is not None and p5_idx is not None and p7_idx is not None:
        ordering_viol = np.mean((x[p3_idx] <= x[p5_idx]) | (x[p5_idx] <= x[p7_idx]))
        diagnostics["ordering"] = {
            "violation_rate": float(ordering_viol),
        }

    return InverseReconstructionResult(
        df_obs=df_obs,
        reconstructed=reconstructed,
        original=original,
        diagnostics=diagnostics,
    )


def make_inverse_reconstruction_series_builder(
    csv_path: str,
    slices: Sequence[Tuple[int, int]],
    flux_types: Sequence[str] = ("p3_flux_ic", "p5_flux_ic", "p7_flux_ic"),
    **kwargs,
):
    """
    Returns a plotting-compatible series builder:
    builder(sample_cf_row, flux_type) -> (df_obs, cf_series, original_series, min_y, max_y)
    """
    cache: MutableMapping[Tuple, InverseReconstructionResult] = {}

    def _key(sample_cfe: pd.Series) -> Tuple:
        # Stable cache key across repeated plot calls for the same CF row.
        return tuple(np.asarray(sample_cfe.values, dtype=object).tolist())

    def builder(sample_cfe: pd.Series, flux_type: str):
        k = _key(sample_cfe)
        if k not in cache:
            cache[k] = inverse_reconstruct_counterfactual_series(
                csv_path=csv_path,
                sample_cfe=sample_cfe,
                slices=slices,
                flux_types=flux_types,
                **kwargs,
            )
        res = cache[k]
        if flux_type not in res.reconstructed:
            raise KeyError(f"Flux '{flux_type}' not in reconstructed channels: {list(res.reconstructed)}")
        cf_series = res.reconstructed[flux_type]
        orig_series = res.original[flux_type]
        min_y = float(min(cf_series.min(), orig_series.min()))
        max_y = float(max(cf_series.max(), orig_series.max()))
        return res.df_obs, cf_series, orig_series, min_y, max_y

    return builder
