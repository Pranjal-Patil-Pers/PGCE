# Quickstart

This quickstart demonstrates a minimal constrained DiCE workflow.

## 1. Prepare training dataframe

Your dataframe must include an outcome column, e.g. `Event_Y_N`.

```python
import pandas as pd

train_df = pd.read_csv("train_with_outcome.csv")
query_df = pd.read_csv("query_rows.csv")
```

## 2. Define constraints

```python
from PGCE import FeatureRangeConstraint, OrderedFeaturesConstraint

constraints = [
    FeatureRangeConstraint(
        ranges={
            "f0": [0.0, 10.0],
            "f1": [0.0, 10.0],
        },
        penalty_value=1e4,
    ),
    OrderedFeaturesConstraint(
        ordered_feature_groups=[("f0", "f1", "f2")],
        increasing=False,
        strict=True,
        penalty_value=1e4,
    ),
]
```

## 3. Build explainer

```python
from PGCE import build_constrained_explainer

explainer, interfaces = build_constrained_explainer(
    dataframe=train_df,
    model=trained_model,
    outcome_name="Event_Y_N",
    constraints=constraints,
)
```

## 4. Generate counterfactuals

```python
from PGCE import generate_counterfactuals, extract_first_counterfactual_df

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
print(cf_df.head())
```

## 5. Plot feature deltas

```python
from PGCE import plot_counterfactual_deltas

plot_counterfactual_deltas(query_df.iloc[[0]], cf_df.drop(columns=["Event_Y_N"]))
```

See the README for a complete usage example.
