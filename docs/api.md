# API Reference

## Core classes

- `ConstrainedDiceGenetic`
  - Extension of DiCE's genetic explainer that adds custom penalty terms.

- `CFEAnalyzer`
  - Utility class for post-hoc analysis and diagnostic summaries.

- `InverseReconstructionResult`
  - Dataclass with reconstructed/original curves and diagnostics.

## Constraints

- `FeatureRangeConstraint`
- `OrderedFeaturesConstraint`
- `FeatureThresholdConstraint`
- `CounterfactualConstraint` protocol
- `evaluate_constraint(...)`

## Pipeline functions

- `build_dice_interfaces(...)`
- `build_constrained_explainer(...)`
- `generate_counterfactuals(...)`
- `extract_counterfactual_dfs(...)`
- `extract_first_counterfactual_df(...)`

## Inverse reconstruction

- `inverse_reconstruct_counterfactual_series(...)`
- `make_inverse_reconstruction_series_builder(...)`

## Plotting

- `plot_counterfactual_deltas(...)`
- `plot_counterfactual_profiles(...)`
- `plot_time_series_counterfactual_grid(...)`
