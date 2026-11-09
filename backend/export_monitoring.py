"""Export holdout curves from the notebook for the Streamlit monitoring page.

Run this from the notebook after `y_hold`, `test_probs` and the cost values exist:

    from backend.export_monitoring import export_monitoring_data
    export_monitoring_data(y_hold, test_probs, threshold=FINAL_THRESHOLD,
                           false_positive_cost=C_FP, false_negative_cost=C_FN)

These curves are descriptive holdout diagnostics only. Do not use them to retune
or reselect the model after evaluating the official holdout set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve


def export_monitoring_data(
    y_true: Any,
    probabilities: Any,
    threshold: float = 0.01811,
    false_positive_cost: float = 10,
    false_negative_cost: float = 500,
    output_path: str | Path | None = None,
) -> Path:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)

    if len(y) != len(p) or len(y) == 0:
        raise ValueError("y_true and probabilities must have the same non-zero length.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite values between 0 and 1.")

    precision, recall, pr_thresholds = precision_recall_curve(y, p)
    fraction_positive, mean_predicted = calibration_curve(
        y, p, n_bins=10, strategy="quantile"
    )

    thresholds = np.unique(np.concatenate([
        np.linspace(0.0005, 0.99, 299),
        np.asarray([threshold], dtype=float),
    ]))
    cost_curve = []
    for candidate in thresholds:
        pred = p >= candidate
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        cost_curve.append({
            "threshold": float(candidate),
            "cost": float(false_positive_cost * fp + false_negative_cost * fn),
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "true_negatives": tn,
            "inspection_rate": float(pred.mean()),
        })

    data = {
        "model_name": "Baseline XGBoost",
        "evaluation_source": "official_holdout_evaluation",
        "holdout_rows": int(len(y)),
        "holdout_positive_cases": int(y.sum()),
        "threshold": float(threshold),
        "probabilities_calibrated": False,
        "cost_function": f"{false_positive_cost} * false_positives + {false_negative_cost} * false_negatives",
        "curves": {
            "precision_recall": {
                "precision": precision.astype(float).tolist(),
                "recall": recall.astype(float).tolist(),
                "thresholds": pr_thresholds.astype(float).tolist(),
            },
            "calibration": {
                "mean_predicted_probability": mean_predicted.astype(float).tolist(),
                "observed_positive_fraction": fraction_positive.astype(float).tolist(),
            },
            "cost_vs_threshold": cost_curve,
        },
        "note": "Holdout curves are descriptive evaluation outputs only. Do not use them to select a new threshold or model.",
    }

    if output_path is None:
        output_path = Path(__file__).resolve().parent / "data" / "model_monitoring.json"
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path
