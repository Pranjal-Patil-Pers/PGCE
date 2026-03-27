# SEP_CFE_DiCE

Standalone Python package for constrained counterfactual generation with DiCE, extracted from `src/SEP_CFE_DiCE` in the Solar Energy Particle Prediction project.

## What this package provides

- `ConstrainedDiceGenetic`: DiCE genetic explainer with pluggable custom penalties.
- Constraint primitives:
  - `FeatureRangeConstraint`
  - `OrderedFeaturesConstraint`
  - `FeatureThresholdConstraint`
- Pipeline helpers for building DiCE interfaces and generating counterfactuals.
- Counterfactual plotting utilities for feature and time-series views.
- Inverse reconstruction utilities for rebuilding time-series curves from window-level counterfactual features.
- `CFEAnalyzer` utilities retained for compatibility with existing notebooks.

## Installation

```bash
pip install -e .
```

For notebook usage:

```bash
pip install -e ".[notebooks]"
```

## Quick usage

```python
import pandas as pd
from SEP_CFE_DiCE import (
    FeatureRangeConstraint,
    OrderedFeaturesConstraint,
    build_constrained_explainer,
    generate_counterfactuals,
    extract_first_counterfactual_df,
)

train_df = pd.read_csv("train_with_outcome.csv")
query_df = pd.read_csv("query_rows.csv")

constraints = [
    FeatureRangeConstraint(ranges={"f0": [0.0, 10.0]}, penalty_value=1e4),
    OrderedFeaturesConstraint(
        ordered_feature_groups=[("f0", "f1", "f2")],
        increasing=False,
        strict=True,
        penalty_value=1e4,
    ),
]

explainer, _ = build_constrained_explainer(
    dataframe=train_df,
    model=trained_model,
    outcome_name="target",
    constraints=constraints,
)

cf_obj = generate_counterfactuals(
    explainer=explainer,
    query_instances=query_df,
    total_cfs=3,
    desired_class=1,
)

cf_df = extract_first_counterfactual_df(cf_obj)
```

Variable guide (one-liners):

- `trained_model`: your trained binary classifier used to score candidate counterfactuals.
- `train_df`: training table with feature columns plus the outcome column (`target`).
- `query_df`: feature-only rows you want counterfactuals for.
- `constraints`: list of constraint objects that penalize invalid counterfactuals.
- `explainer`: configured constrained DiCE explainer used for optimization.
- `cf_obj`: raw DiCE result object containing generated counterfactual sets.
- `cf_df`: first counterfactual set from `cf_obj`, flattened into a pandas DataFrame.

## Example notebook

Open:

- `examples/sep_cfe_dice_quickstart.ipynb`

The notebook shows:

1. Building synthetic tabular training/query data.
2. Training a sklearn model.
3. Defining and applying constraints.
4. Generating and plotting counterfactuals.

## Documentation

- `docs/installation.md`
- `docs/quickstart.md`
- `docs/api.md`
- `docs/development.md`

## Development

```bash
git clone <your-new-repo-url>
cd sep-cfe-dice
pip install -e ".[dev,notebooks]"
pytest
```

## License

MIT (see `LICENSE`).
