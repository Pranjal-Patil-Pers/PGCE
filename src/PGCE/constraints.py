from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Protocol, Sequence, Union

import numpy as np

DesiredClass = Union[int, float, str, None]
ConstraintResult = Union[float, int, np.ndarray]


class CounterfactualConstraint(Protocol):
    def penalty(self, candidates: np.ndarray, feature_to_index: Mapping[str, int], query_instance: Optional[np.ndarray] = None, desired_class: DesiredClass = None) -> np.ndarray:
        """Returns a penalty for each candidate (higher is worse)."""


ConstraintCallable = Callable[[np.ndarray, Mapping[str, int], Optional[np.ndarray], DesiredClass], ConstraintResult]
ConstraintLike = Union[CounterfactualConstraint, ConstraintCallable]


def _as_penalty_vector(raw_penalty: ConstraintResult, n_candidates: int) -> np.ndarray:
    penalty = np.asarray(raw_penalty, dtype=float)
    if penalty.ndim == 0:
        return np.full(n_candidates, float(penalty), dtype=float)
    penalty = penalty.reshape(-1)
    if penalty.shape[0] != n_candidates:
        raise ValueError(f"Constraint returned {penalty.shape[0]} penalties for {n_candidates} candidates.")
    return penalty


def evaluate_constraint(constraint: ConstraintLike, candidates: np.ndarray, feature_to_index: Mapping[str, int], query_instance: Optional[np.ndarray] = None, desired_class: DesiredClass = None) -> np.ndarray:
    if hasattr(constraint, "penalty"):
        raw_penalty = constraint.penalty(candidates=candidates, feature_to_index=feature_to_index, query_instance=query_instance, desired_class=desired_class)
    elif callable(constraint):
        raw_penalty = constraint(candidates, feature_to_index, query_instance, desired_class)
    else:
        raise TypeError("Constraint must either expose .penalty(...) or be a callable with signature (candidates, feature_to_index, query_instance, desired_class).")
    return _as_penalty_vector(raw_penalty, candidates.shape[0])


@dataclass(frozen=True)
class FeatureRangeConstraint:
    ranges: Mapping[str, Sequence[float]]
    penalty_value: float = 1e4
    strict_bounds: bool = False

    def penalty(self, candidates: np.ndarray, feature_to_index: Mapping[str, int], query_instance: Optional[np.ndarray] = None, desired_class: DesiredClass = None) -> np.ndarray:
        penalties = np.zeros(candidates.shape[0], dtype=float)
        for feature_name, bounds in self.ranges.items():
            idx = feature_to_index.get(feature_name)
            if idx is None:
                continue
            lower, upper = float(bounds[0]), float(bounds[1])
            values = candidates[:, idx]
            if self.strict_bounds:
                out_of_bounds = (values <= lower) | (values >= upper)
            else:
                out_of_bounds = (values < lower) | (values > upper)
            penalties += out_of_bounds.astype(float) * self.penalty_value
        return penalties


@dataclass(frozen=True)
class OrderedFeaturesConstraint:
    ordered_feature_groups: Sequence[Sequence[str]]
    penalty_value: float = 1e4
    increasing: bool = False
    strict: bool = True
    skip_missing: bool = True

    def penalty(self, candidates: np.ndarray, feature_to_index: Mapping[str, int], query_instance: Optional[np.ndarray] = None, desired_class: DesiredClass = None) -> np.ndarray:
        penalties = np.zeros(candidates.shape[0], dtype=float)
        for group in self.ordered_feature_groups:
            indices = []
            missing = []
            for feature_name in group:
                idx = feature_to_index.get(feature_name)
                if idx is None:
                    missing.append(feature_name)
                else:
                    indices.append(idx)
            if missing:
                if self.skip_missing:
                    continue
                raise ValueError(f"Missing features for order constraint: {missing}")
            if len(indices) < 2:
                continue
            for left_idx, right_idx in zip(indices[:-1], indices[1:]):
                left_values = candidates[:, left_idx]
                right_values = candidates[:, right_idx]
                if self.increasing:
                    violations = left_values >= right_values if self.strict else left_values > right_values
                else:
                    violations = left_values <= right_values if self.strict else left_values < right_values
                penalties += violations.astype(float) * self.penalty_value
        return penalties


@dataclass(frozen=True)
class FeatureThresholdConstraint:
    feature_name: str
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    penalty_value: float = 1e4

    def __post_init__(self) -> None:
        if self.lower_bound is None and self.upper_bound is None:
            raise ValueError("At least one bound must be set for FeatureThresholdConstraint.")

    def penalty(self, candidates: np.ndarray, feature_to_index: Mapping[str, int], query_instance: Optional[np.ndarray] = None, desired_class: DesiredClass = None) -> np.ndarray:
        penalties = np.zeros(candidates.shape[0], dtype=float)
        idx = feature_to_index.get(self.feature_name)
        if idx is None:
            return penalties
        values = candidates[:, idx]
        violations = np.zeros(candidates.shape[0], dtype=bool)
        if self.lower_bound is not None:
            violations |= values < self.lower_bound if self.lower_inclusive else values <= self.lower_bound
        if self.upper_bound is not None:
            violations |= values > self.upper_bound if self.upper_inclusive else values >= self.upper_bound
        penalties += violations.astype(float) * self.penalty_value
        return penalties
