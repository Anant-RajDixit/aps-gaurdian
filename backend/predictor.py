from __future__ import annotations

import inspect
import io
import os
from pathlib import Path
from typing import Any

import cloudpickle
import numpy as np
import pandas as pd


class ModelArtifactError(RuntimeError):
    """Raised when the saved model bundle cannot be loaded or inspected."""


class InputValidationError(ValueError):
    """Raised when uploaded sensor data does not match the model schema."""


class APSPredictor:
    """Loads the saved notebook artifact and provides batch/single scoring."""

    ID_COLUMNS = ("vehicle_id", "truck_id", "record_id", "id")
    TARGET_COLUMN = "class"

    def __init__(self, artifact_path: str | Path):
        self.artifact_path = Path(artifact_path).expanduser().resolve()
        if not self.artifact_path.is_file():
            raise ModelArtifactError(
                f"Model artifact not found: {self.artifact_path}. "
                "Copy scania_aps_xgboost_final.pkl into the project's model/ folder."
            )

        try:
            with self.artifact_path.open("rb") as file:
                artifact = cloudpickle.load(file)
        except Exception as exc:  # pickle loading errors can vary by dependency/version
            raise ModelArtifactError(
                "Could not load the model artifact. Check that the artifact is trusted "
                "and that the installed scikit-learn, XGBoost and Cloudpickle versions "
                "are compatible with the environment used to save it."
            ) from exc

        if not isinstance(artifact, dict) or "pipeline" not in artifact:
            raise ModelArtifactError(
                "The artifact must be a dictionary containing a 'pipeline' entry."
            )

        self.artifact = artifact
        self.pipeline = artifact["pipeline"]
        self.threshold = float(artifact.get("threshold", 0.01811))
        self.model_name = str(artifact.get("model_name", "Baseline XGBoost"))
        self.feature_names = list(artifact.get("feature_names", []))
        self.false_positive_cost = float(artifact.get("false_positive_cost", 10))
        self.false_negative_cost = float(artifact.get("false_negative_cost", 500))

        if not 0 < self.threshold < 1:
            raise ModelArtifactError("The saved classification threshold must be between 0 and 1.")
        if not hasattr(self.pipeline, "predict_proba"):
            raise ModelArtifactError("The saved pipeline does not implement predict_proba().")

        try:
            self.preprocessor = self.pipeline.named_steps["preprocessor"]
            self.classifier = self.pipeline.named_steps["classifier"]
        except Exception as exc:
            raise ModelArtifactError(
                "Expected a pipeline with steps named 'preprocessor' and 'classifier'."
            ) from exc

        self.required_features = self._extract_raw_feature_names()
        if not self.required_features:
            raise ModelArtifactError(
                "Could not recover the original raw sensor-column names from the saved "
                "feature-engineering function. The artifact may have been saved from a "
                "different pipeline structure."
            )

        self._shap_explainer = None

        model_feature_count = getattr(self.classifier, "n_features_in_", None)
        if model_feature_count is not None and self.feature_names:
            if int(model_feature_count) != len(self.feature_names):
                raise ModelArtifactError(
                    "Saved feature_names do not match the classifier's input feature count."
                )

    def _extract_raw_feature_names(self) -> list[str]:
        """Recover the original input columns from the fitted preprocessing step.

        The notebook's feature-engineering function captures the raw schema in a
        closure. That closure is useful when available, but is an implementation
        detail of cloudpickle and can deserialize differently across Python or
        scikit-learn versions. ``feature_names_in_`` is persisted by the fitted
        FunctionTransformer and is therefore the reliable fallback for deployed
        artifacts.
        """
        feature_step = None
        try:
            feature_step = self.preprocessor.named_steps["feature_engineering"]
            engineer_function = feature_step.func
            closure_values = inspect.getclosurevars(engineer_function).nonlocals
            raw_columns = closure_values.get("original_feature_columns")
            if raw_columns is None:
                # Fallback for callable function objects whose closure inspection differs.
                freevars = engineer_function.__code__.co_freevars
                cells = engineer_function.__closure__ or ()
                cell_values = dict(zip(freevars, (cell.cell_contents for cell in cells)))
                raw_columns = cell_values.get("original_feature_columns")
            if raw_columns is not None:
                return [str(column) for column in raw_columns]
        except Exception:
            # A cloudpickled nested function can have a closure that is not
            # introspectable in another runtime. Try persisted fitted metadata.
            pass

        for fitted_object in (feature_step, self.preprocessor, self.pipeline):
            raw_columns = getattr(fitted_object, "feature_names_in_", None)
            if raw_columns is not None and len(raw_columns):
                return [str(column) for column in raw_columns]

        return []

    def model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "algorithm": "XGBoost",
            "threshold": self.threshold,
            "threshold_source": "minimum OOF misclassification cost",
            "false_positive_cost": self.false_positive_cost,
            "false_negative_cost": self.false_negative_cost,
            "raw_input_feature_count": len(self.required_features),
            "engineered_feature_count": len(self.feature_names),
            "positive_class": 1,
            "positive_class_label": "APS failure",
            "probabilities_calibrated": False,
            "score_note": "Model scores are uncalibrated; use the saved threshold for the inspection decision.",
        }

    def validate_frame(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
        if not isinstance(frame, pd.DataFrame):
            raise InputValidationError("Input must be a pandas DataFrame.")
        if frame.empty:
            raise InputValidationError("The uploaded dataset contains no data rows.")
        if frame.columns.duplicated().any():
            duplicate_names = frame.columns[frame.columns.duplicated()].astype(str).tolist()
            raise InputValidationError(f"Duplicate column names found: {duplicate_names[:10]}")

        original = frame.copy()
        id_column = next((col for col in self.ID_COLUMNS if col in original.columns), None)
        if id_column:
            raw_ids = original[id_column].astype("string").fillna("").tolist()
            record_ids = [value if value.strip() else f"Truck-{i + 1:06d}" for i, value in enumerate(raw_ids)]
        else:
            record_ids = [f"Truck-{i + 1:06d}" for i in range(len(original))]

        # A target column may be present in the official UCI CSV. Never pass it to the model.
        target_was_ignored = self.TARGET_COLUMN in original.columns

        missing_columns = [col for col in self.required_features if col not in original.columns]
        if missing_columns:
            preview = missing_columns[:25]
            suffix = " ..." if len(missing_columns) > len(preview) else ""
            raise InputValidationError(
                f"Missing {len(missing_columns)} required sensor columns: {preview}{suffix}. "
                "Upload the original APS feature schema (169 sensor columns); the optional 'class' target is ignored."
            )

        metadata_columns = set(self.ID_COLUMNS) | {self.TARGET_COLUMN}
        extra_columns = [
            str(col) for col in original.columns
            if col not in self.required_features and col not in metadata_columns
        ]

        # Preserve the exact feature order used when training the model.
        raw_features = original.loc[:, self.required_features].copy()
        invalid_numeric_cells = 0
        for column in self.required_features:
            source = raw_features[column]
            converted = pd.to_numeric(source, errors="coerce")
            invalid_numeric_cells += int((source.notna() & converted.isna()).sum())
            raw_features[column] = converted

        raw_features = raw_features.replace([np.inf, -np.inf], np.nan)
        missing_per_row = raw_features.isna().sum(axis=1)
        missing_cells = int(raw_features.isna().sum().sum())
        all_missing_columns = raw_features.columns[raw_features.isna().all()].tolist()
        quality = {
            "rows": int(len(raw_features)),
            "required_feature_count": int(len(self.required_features)),
            "missing_sensor_cells": missing_cells,
            "missing_sensor_rate": float(missing_cells / (len(raw_features) * len(self.required_features))),
            "rows_with_missing_sensors": int((missing_per_row > 0).sum()),
            "max_missing_features_in_one_row": int(missing_per_row.max()) if len(missing_per_row) else 0,
            "invalid_numeric_cells_coerced_to_missing": invalid_numeric_cells,
            "all_missing_columns": all_missing_columns,
            "ignored_extra_columns": extra_columns,
            "target_column_ignored": target_was_ignored,
        }
        raw_features.attrs["missing_sensor_count"] = missing_per_row.to_numpy()
        return raw_features, record_ids, quality

    def score_frame(self, frame: pd.DataFrame) -> dict[str, Any]:
        raw_features, record_ids, quality = self.validate_frame(frame)
        try:
            scores = self.pipeline.predict_proba(raw_features)[:, 1]
        except Exception as exc:
            raise RuntimeError(
                "The saved pipeline could not transform or score this input. "
                "Check that the CSV schema and installed ML library versions match the training environment."
            ) from exc

        predictions = scores >= self.threshold
        missing_counts = raw_features.attrs.get("missing_sensor_count", np.zeros(len(raw_features), dtype=int))

        results_frame = pd.DataFrame({
            "record_id": record_ids,
            "aps_failure_score": scores.astype(float),
            "prediction": predictions.astype(int),
            "decision": np.where(predictions, "INSPECT", "NO_INSPECTION"),
            "priority": np.where(predictions, "INSPECTION", "ROUTINE"),
            "missing_sensor_count": np.asarray(missing_counts, dtype=int),
        })
        results_frame = results_frame.sort_values(
            "aps_failure_score", ascending=False, kind="stable"
        ).reset_index(drop=True)

        flagged = int(predictions.sum())
        total = int(len(predictions))
        return {
            "model_name": self.model_name,
            "score_type": "uncalibrated_model_score",
            "threshold": self.threshold,
            "summary": {
                "total_vehicles": total,
                "vehicles_recommended_for_inspection": flagged,
                "vehicles_not_flagged": total - flagged,
                "inspection_rate": float(flagged / total) if total else 0.0,
                "mean_model_score": float(np.mean(scores)) if total else 0.0,
            },
            "data_quality": quality,
            "results": results_frame.to_dict(orient="records"),
        }

    def validate_single(self, features: dict[str, Any]) -> pd.DataFrame:
        if not isinstance(features, dict):
            raise InputValidationError("'features' must be a JSON object.")
        return pd.DataFrame([features])

    def explain_single(self, features: dict[str, Any], record_id: str | None = None, top_n: int = 10) -> dict[str, Any]:
        try:
            import shap
        except ImportError as exc:
            raise RuntimeError("SHAP is not installed in the backend environment. Install the backend requirements.") from exc

        frame = self.validate_single(features)
        raw_features, record_ids, quality = self.validate_frame(frame)
        transformed = self.preprocessor.transform(raw_features)
        transformed_frame = pd.DataFrame(transformed, columns=self.feature_names)
        score = float(self.classifier.predict_proba(transformed)[:, 1][0])
        prediction = int(score >= self.threshold)

        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.classifier)

        shap_values = self._shap_explainer.shap_values(transformed_frame)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            shap_array = shap_array[0, :, 1]
        else:
            shap_array = shap_array[0]

        expected_value = np.asarray(self._shap_explainer.expected_value)
        if expected_value.ndim > 0 and expected_value.size > 1:
            base_value = float(expected_value.reshape(-1)[1])
        else:
            base_value = float(expected_value.reshape(-1)[0])

        top_n = max(1, min(int(top_n), 25))
        order = np.argsort(np.abs(shap_array))[::-1][:top_n]
        contributions = []
        for idx in order:
            value = float(shap_array[idx])
            contributions.append({
                "feature": self.feature_names[int(idx)],
                "shap_value": value,
                "direction": "pushes_towards_aps_failure" if value > 0 else "pushes_away_from_aps_failure",
            })

        return {
            "record_id": record_id or record_ids[0],
            "aps_failure_score": score,
            "threshold": self.threshold,
            "decision": "INSPECT" if prediction else "NO_INSPECTION",
            "explanation_scale": "raw_model_output_log_odds",
            "base_value": base_value,
            "top_contributions": contributions,
            "data_quality": quality,
            "note": "SHAP contributions explain model output, not causal effects or percentage-point changes in failure probability.",
        }
