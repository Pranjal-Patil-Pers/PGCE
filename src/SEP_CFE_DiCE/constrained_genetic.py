from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from dice_ml.explainer_interfaces.dice_genetic import DiceGenetic

from .constraints import ConstraintLike, evaluate_constraint


class ConstrainedDiceGenetic(DiceGenetic):
    """
    Data-agnostic extension of DiCE Genetic explainer with pluggable constraints.

    This implementation is intentionally independent of any specific dataset schema.
    You pass feature constraints as reusable objects/callables.
    """

    def __init__(
        self,
        data_interface,
        model_interface,
        constraints: Optional[Sequence[ConstraintLike]] = None,
        l0_penalty_weight: float = 0.0,
    ):
        super().__init__(data_interface, model_interface)
        self.constraints: List[ConstraintLike] = list(constraints or [])
        self.l0_penalty_weight = float(l0_penalty_weight)
        self._feature_to_index_cache: Optional[Dict[str, int]] = None

    def add_constraint(self, constraint: ConstraintLike) -> None:
        self.constraints.append(constraint)

    def clear_constraints(self) -> None:
        self.constraints.clear()

    def _feature_to_index(self) -> Dict[str, int]:
        if self._feature_to_index_cache is None:
            self._feature_to_index_cache = {
                name: idx for idx, name in enumerate(self.data_interface.feature_names)
            }
        return self._feature_to_index_cache

    def compute_constraint_penalty(self, candidates: np.ndarray, desired_class) -> np.ndarray:
        if not self.constraints:
            return np.zeros(candidates.shape[0], dtype=float)

        feature_to_index = self._feature_to_index()
        query_instance = getattr(self, "x1", None)
        penalties = np.zeros(candidates.shape[0], dtype=float)
        for constraint in self.constraints:
            penalties += evaluate_constraint(
                constraint=constraint,
                candidates=candidates,
                feature_to_index=feature_to_index,
                query_instance=query_instance,
                desired_class=desired_class,
            )
        return penalties

    def compute_l0_penalty(self, candidates: np.ndarray) -> np.ndarray:
        if not hasattr(self, "x1"):
            return np.zeros(candidates.shape[0], dtype=float)
        query_instance = np.asarray(self.x1).reshape(1, -1)
        changed = np.count_nonzero(candidates != query_instance, axis=1)
        return changed / candidates.shape[1]

    def compute_diversity_loss(self, candidates: np.ndarray) -> np.ndarray:
        """
        Mean pairwise L1 distance among candidates.

        Returning the same value for all candidates keeps compatibility with
        the population-level loss structure in DiCE Genetic.
        """
        n_candidates = candidates.shape[0]
        if n_candidates < 2:
            return np.zeros(n_candidates, dtype=float)

        pairwise = np.abs(candidates[:, None, :] - candidates[None, :, :]).sum(axis=2)
        tri_upper = pairwise[np.triu_indices(n_candidates, k=1)]
        mean_diversity = float(np.mean(tri_upper)) if tri_upper.size else 0.0
        return np.full(n_candidates, mean_diversity, dtype=float)

    def compute_loss(self, cfs: np.ndarray, desired_range, desired_class):  # noqa: D401
        """Computes DiCE loss + custom penalties/regularizers."""
        self.yloss = self.compute_yloss(cfs, desired_range, desired_class)
        self.proximity_loss = (
            self.compute_proximity_loss(cfs, self.query_instance_normalized)
            if self.proximity_weight > 0
            else np.zeros(cfs.shape[0], dtype=float)
        )
        self.sparsity_loss = (
            self.compute_sparsity_loss(cfs)
            if self.sparsity_weight > 0
            else np.zeros(cfs.shape[0], dtype=float)
        )
        self.diversity_loss = (
            self.compute_diversity_loss(cfs)
            if self.diversity_weight > 0
            else np.zeros(cfs.shape[0], dtype=float)
        )

        self.constraint_penalty = self.compute_constraint_penalty(cfs, desired_class=desired_class)
        self.l0_penalty = (
            self.compute_l0_penalty(cfs) * self.l0_penalty_weight
            if self.l0_penalty_weight > 0
            else np.zeros(cfs.shape[0], dtype=float)
        )

        total_loss = (
            np.asarray(self.yloss, dtype=float)
            + (self.proximity_weight * np.asarray(self.proximity_loss, dtype=float))
            + (self.sparsity_weight * np.asarray(self.sparsity_loss, dtype=float))
            - (self.diversity_weight * np.asarray(self.diversity_loss, dtype=float))
            + np.asarray(self.constraint_penalty, dtype=float)
            + np.asarray(self.l0_penalty, dtype=float)
        )

        index = np.arange(len(cfs), dtype=float)
        self.loss = np.column_stack([index, total_loss])
        return self.loss
