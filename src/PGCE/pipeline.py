from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from .constrained_genetic import ConstrainedDiceGenetic
from .constraints import ConstraintLike


@dataclass(frozen=True)
class DiceInterfaces:
    data_interface: object
    model_interface: object


def build_dice_interfaces(
    dataframe: pd.DataFrame,
    model,
    outcome_name: str,
    continuous_features: Optional[Sequence[str]] = None,
    backend: str = "sklearn",
    model_type: str = "classifier",
) -> DiceInterfaces:
    """
    Build DiCE data/model interfaces from generic tabular data and a trained model.
    """
    import dice_ml

    if outcome_name not in dataframe.columns:
        raise ValueError(f"Outcome column '{outcome_name}' not present in dataframe.")

    if continuous_features is None:
        continuous_features = [col for col in dataframe.columns if col != outcome_name]

    data_interface = dice_ml.Data(
        dataframe=dataframe,
        continuous_features=list(continuous_features),
        outcome_name=outcome_name,
    )
    model_interface = dice_ml.Model(
        model=model,
        backend=backend,
        model_type=model_type,
    )
    return DiceInterfaces(data_interface=data_interface, model_interface=model_interface)


def build_constrained_explainer(
    dataframe: pd.DataFrame,
    model,
    outcome_name: str,
    constraints: Optional[Sequence[ConstraintLike]] = None,
    continuous_features: Optional[Sequence[str]] = None,
    backend: str = "sklearn",
    model_type: str = "classifier",
    l0_penalty_weight: float = 0.0,
) -> Tuple[ConstrainedDiceGenetic, DiceInterfaces]:
    """
    Create a standalone constrained genetic DiCE explainer from data+model+constraints.
    """
    interfaces = build_dice_interfaces(
        dataframe=dataframe,
        model=model,
        outcome_name=outcome_name,
        continuous_features=continuous_features,
        backend=backend,
        model_type=model_type,
    )
    explainer = ConstrainedDiceGenetic(
        data_interface=interfaces.data_interface,
        model_interface=interfaces.model_interface,
        constraints=constraints,
        l0_penalty_weight=l0_penalty_weight,
    )
    return explainer, interfaces


def generate_counterfactuals(
    explainer: ConstrainedDiceGenetic,
    query_instances: pd.DataFrame,
    total_cfs: int,
    desired_class="opposite",
    **kwargs,
):
    """
    Thin wrapper around DiCE generation with consistent parameter naming.
    """
    return explainer.generate_counterfactuals(
        query_instances=query_instances,
        total_CFs=total_cfs,
        desired_class=desired_class,
        **kwargs,
    )


def extract_counterfactual_dfs(counterfactual_explanations) -> List[pd.DataFrame]:
    """
    Extract final counterfactual dataframes (one per query instance).
    """
    output = []
    for cf_example in counterfactual_explanations.cf_examples_list:
        if cf_example.final_cfs_df is None:
            output.append(pd.DataFrame())
        else:
            output.append(cf_example.final_cfs_df.copy())
    return output


def extract_first_counterfactual_df(counterfactual_explanations) -> pd.DataFrame:
    """
    Convenience helper for single-query use cases.
    """
    dfs = extract_counterfactual_dfs(counterfactual_explanations)
    return dfs[0] if dfs else pd.DataFrame()
