# SEP_CFE_DiCE Library API

This module exposes a standalone constrained DiCE genetic framework that accepts:

- tabular training data (with outcome column)
- any trained compatible model
- pluggable constraints

## Minimal usage

```python
import pandas as pd
from SEP_CFE_DiCE import (
    FeatureRangeConstraint,
    OrderedFeaturesConstraint,
    build_constrained_explainer,
    generate_counterfactuals,
    extract_first_counterfactual_df,
    plot_counterfactual_deltas,
    plot_time_series_counterfactual_grid,
)

# dataframe must include outcome column
train_df = pd.read_csv("train_with_outcome.csv")
query_df = pd.read_csv("query_rows.csv")

constraints = [
    FeatureRangeConstraint(ranges=permitted_ranges, penalty_value=1e4),
    OrderedFeaturesConstraint(
        ordered_feature_groups=[("p3_flux_ic_mean@[0:10]", "p5_flux_ic_mean@[0:10]", "p7_flux_ic_mean@[0:10]")],
        increasing=False,
        strict=True,
        penalty_value=1e4,
    ),
]

explainer, _ = build_constrained_explainer(
    dataframe=train_df,
    model=trained_model,
    outcome_name="Event_Y_N",
    constraints=constraints,
)

cf_obj = generate_counterfactuals(
    explainer=explainer,
    query_instances=query_df,
    total_cfs=5,
    desired_class=1,
    proximity_weight=0.2,
    sparsity_weight=0.2,
    diversity_weight=5.0,
)

cf_df = extract_first_counterfactual_df(cf_obj)
plot_counterfactual_deltas(query_df.iloc[[0]], cf_df.drop(columns=["Event_Y_N"]))

# Optional: use your existing reconstruction function from the notebook
# def build_series(sample_cf_row, flux_type): ...
# plot_time_series_counterfactual_grid(cf_df, series_builder=build_series)
```
