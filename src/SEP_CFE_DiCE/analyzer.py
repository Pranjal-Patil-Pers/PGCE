from __future__ import annotations

import re
from collections import Counter
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


class CFEAnalyzer:
    """
    Utility helpers for inspecting generated counterfactual explanations.

    This class is included in this standalone package to preserve compatibility
    with existing project notebooks that imported `CFEAnalyzer` from
    `SEP_CFE_DiCE`.
    """

    def __init__(
        self,
        model,
        feature_names: List[str],
        min_max_dict: dict | None = None,
        window_pattern: str = r"\[(.*?)\]",
    ):
        self.model = model
        self.feature_names = feature_names
        self.min_max_dict = min_max_dict
        self.window_pattern = window_pattern

    def get_feature_importance(self, model=None, feature_names=None) -> pd.DataFrame:
        feature_importances = self.model.feature_importances_
        weight_vector = feature_importances / np.sum(feature_importances)
        normalized_weights_dict = dict(zip(self.feature_names, weight_vector))
        sorted_normalized_weights_dict = dict(
            sorted(normalized_weights_dict.items(), key=lambda item: item[1], reverse=True)
        )
        return pd.DataFrame(
            list(sorted_normalized_weights_dict.items()),
            columns=["Feature", "Normalized Importance"],
        )

    def get_query_instance(self, query_ts_filename, df_combined_labels, model=None):
        """
        Retrieve the query instance and its target label for a given filename.
        """
        query_instance_raw = df_combined_labels[df_combined_labels["File"] == query_ts_filename]
        query_instance = query_instance_raw.drop(
            ["Label", "Event_Y_N", "Multi_Label", "File"], axis=1
        )

        true_label = query_instance_raw["Event_Y_N"].values[0]
        predicted_label = self.model.predict(query_instance)

        print("Target value for the query instance:\n", true_label)
        print("Predicted value:\n", predicted_label)

        return query_instance, true_label, predicted_label

    def extract_slices_from_headers(self, header_list, pattern, target_metric):
        """
        Given a list of column names and a regex pattern, extract slices for a metric.
        """
        result = []
        for col in header_list:
            match = re.match(pattern, col)
            if match:
                metric = match.group(1)
                slice_str = match.group(2)
                start, end = map(int, slice_str.split(":"))
                result.append((metric, start, end))

        filtered = [(start, end) for metric, start, end in result if metric == target_metric]
        filtered.sort(key=lambda x: (x[0], x[1]))
        return filtered

    def extract_feature_ranges(self, csv_path, delim, slices, top_k_features, flux_types):
        """
        Extract min-max ranges for selected time slices from a CSV time-series.
        """
        df = pd.read_csv(csv_path, delimiter=delim)
        df = df.rename(columns={"time_tag": "time_stamp"})
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], format="%Y-%m-%d %H:%M:%S")

        event_start = df["time_stamp"].iloc[0] + timedelta(minutes=300)
        event_end = df["time_stamp"].iloc[0] + timedelta(minutes=660)

        df_obs = df[(df["time_stamp"] >= event_start) & (df["time_stamp"] < event_end)].copy()
        df_obs["minutes"] = (df_obs["time_stamp"] - event_start).dt.total_seconds() / 60

        range_dict = {}
        for flux in flux_types:
            for start_min, end_min in slices:
                slice_data = df_obs[
                    (df_obs["minutes"] >= start_min) & (df_obs["minutes"] < end_min)
                ]
                if slice_data.empty:
                    continue
                key = f"{flux}_mean@[{start_min}:{end_min}]"
                range_dict[key] = [slice_data[flux].min(), slice_data[flux].max()]

        return {k: v for k, v in range_dict.items() if k in top_k_features}

    def get_pertubed_series(
        self,
        csv_path,
        sample_cfe,
        flux_type,
        slices,
        start_offset_min=300,
        end_offset_min=660,
    ):
        df = pd.read_csv(csv_path, delimiter=",")
        df = df.rename(columns={"time_tag": "time_stamp"})
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], format="%Y-%m-%d %H:%M:%S")
        event_start = df["time_stamp"].iloc[0] + timedelta(minutes=start_offset_min)
        event_end = df["time_stamp"].iloc[0] + timedelta(minutes=end_offset_min)
        df_obs = df[(df["time_stamp"] >= event_start) & (df["time_stamp"] < event_end)].copy()
        df_obs["minutes"] = (df_obs["time_stamp"] - event_start).dt.total_seconds() / 60

        offset_accum = np.zeros_like(df_obs["minutes"], dtype=float)
        offset_count = np.zeros_like(df_obs["minutes"], dtype=int)

        for start_min, end_min in slices:
            slice_data = df_obs[
                (df_obs["minutes"] >= start_min) & (df_obs["minutes"] < end_min)
            ]
            if slice_data.empty:
                continue
            mask = (df_obs["minutes"] >= start_min) & (df_obs["minutes"] < end_min)
            if np.sum(mask) == 0:
                continue
            flux_data = slice_data[flux_type].values
            pattern = f"^{flux_type}_mean@\\[{start_min}:{end_min}\\]$"
            cfe_value = sample_cfe.filter(regex=pattern).iloc[0]
            global_adjustment = cfe_value - flux_data.mean()
            delta = flux_data + global_adjustment
            offset_accum[mask] += delta
            offset_count[mask] += 1

        final_offset = np.zeros_like(df_obs["minutes"], dtype=float)
        nonzero = offset_count > 0
        final_offset[nonzero] = offset_accum[nonzero] / offset_count[nonzero]

        original = df_obs[flux_type].values
        final_series = original + final_offset
        perturbed_series = pd.Series(final_series, index=df_obs.index)
        original_series = pd.Series(original, index=df_obs.index)
        min_y = min(final_series.min(), final_offset.min(), original.min())
        max_y = max(final_series.max(), final_offset.max(), original.max())
        return df_obs, perturbed_series, original_series, min_y, max_y

    def get_tss(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return tp / (tp + fn) - fp / (fp + tn)

    def get_hss(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return 2 * (tp * tn - fp * fn) / (
            (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
        )

    def analyze_counterfactuals(
        self,
        df: pd.DataFrame,
        min_max_dict: dict,
        query_instance,
        window_pattern=r"\[(.*?)\]",
    ):
        """
        Analyze min/max violations and change frequencies in a CF dataframe.
        """
        effective_ranges = min_max_dict if min_max_dict is not None else self.min_max_dict
        if effective_ranges is None:
            raise ValueError("No min/max dictionary provided. Pass min_max_dict or set self.min_max_dict.")

        violations = []
        for col, (min_val, max_val) in effective_ranges.items():
            if col in df.columns:
                out_of_range = (df[col] < min_val) | (df[col] > max_val)
                if out_of_range.any():
                    violating_rows = df.index[out_of_range].tolist()
                    violations.append(
                        {
                            "column": col,
                            "rows": violating_rows,
                            "min": min_val,
                            "max": max_val,
                        }
                    )

        if isinstance(query_instance, pd.DataFrame):
            query_instance = query_instance.iloc[0]
        query_instance = query_instance.reindex(df.columns)
        changed = df.ne(query_instance, axis=1)
        feature_change_counts = changed.sum(axis=0)
        changed_features = feature_change_counts[feature_change_counts > 0]
        changed_features_sorted = changed_features.sort_values(ascending=False)
        features_sorted_by_change = changed_features_sorted.index.tolist()

        features_for_window = [
            feat
            for feat in features_sorted_by_change
            if isinstance(feat, str) and feat != "Event_Y_N"
        ]

        windows = []
        for feat in features_for_window:
            match = re.search(self.window_pattern, feat)
            if match:
                windows.append(match.group(1))
        window_counts = Counter(windows)
        sorted_windows = window_counts.most_common()

        return {
            "violations": violations,
            "changed_features_ordered": changed_features_sorted,
            "changed_windows_ordered": sorted_windows,
        }
