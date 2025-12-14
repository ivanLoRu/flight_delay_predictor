from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request


# Keep TensorFlow modest on thread usage for local runs
# and enable XLA where available.
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.optimizer.set_jit(True)

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
WINDOW = 12

FEATURE_COLUMNS: List[str] = [
    "Arrival Delay (Minutes)",
    "temp",
    "dwpt",
    "rhum",
    "prcp",
    "wdir",
    "wspd",
    "pres",
    "monthly_passenger_arr",
    "monthly_freight_arr",
    "total_arr",
    "monthly_seats_arr",
    "monthly_passenger_dep",
    "monthly_freight_dep",
    "total_dep",
    "monthly_seats_dep",
    "load_factor_arr",
    "load_factor_dep",
    "dep_delay_mean3h",
    "dep_delay_mean6h",
    "dep_delay_mean12h",
    "dep_delay_count3h",
    "dep_delay_count6h",
    "dep_delay_count12h",
    "dep_last_1h",
    "dep_last_3h",
    "bank_density_15min",
    "turnaround_slack",
]

_MODEL_CACHE: Dict[str, object] = {}


def load_assets() -> Dict[str, object]:
    """Load models and scalers once and reuse them across requests."""
    if _MODEL_CACHE:
        return _MODEL_CACHE

    paths = {
        "clf_model": MODELS_DIR / "delay_clf_lstm.keras",
        "reg_model": MODELS_DIR / "delay_reg_lstm.keras",
        "x_scaler": MODELS_DIR / "reg_x_scaler.pkl",
        "y_scaler": MODELS_DIR / "reg_y_scaler.pkl",
    }
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model assets: {', '.join(missing)}")

    assets = {
        "clf_model": tf.keras.models.load_model(paths["clf_model"]),
        "reg_model": tf.keras.models.load_model(paths["reg_model"]),
        "x_scaler": joblib.load(paths["x_scaler"]),
        "y_scaler": joblib.load(paths["y_scaler"]),
    }
    _MODEL_CACHE.update(assets)
    return assets


def build_sequences(matrix: np.ndarray) -> np.ndarray:
    """Repeat each feature row to satisfy the LSTM window length."""
    return np.repeat(matrix[:, np.newaxis, :], WINDOW, axis=1)


def run_predictions(df: pd.DataFrame, assets: Dict[str, object]):
    """Run classifier and regressor for every row in the dataframe."""
    feature_df = df[FEATURE_COLUMNS].astype(float)
    matrix = feature_df.to_numpy(dtype=np.float32)

    # Classifier
    clf_seq = build_sequences(matrix)
    clf_probs = assets["clf_model"].predict(clf_seq, verbose=0).flatten()
    clf_labels = np.where(clf_probs >= 0.5, "Delay", "On Time")

    # Regressor
    scaled = assets["x_scaler"].transform(matrix)
    reg_seq = build_sequences(scaled)
    reg_scaled = assets["reg_model"].predict(reg_seq, verbose=0).flatten()
    reg_minutes = assets["y_scaler"].inverse_transform(reg_scaled.reshape(-1, 1)).flatten()

    rows = []
    for idx, (label, prob, minutes) in enumerate(zip(clf_labels, clf_probs, reg_minutes), start=1):
        rows.append(
            {
                "row": idx,
                "class_label": str(label),
                "class_prob": float(prob),
                "delay_minutes": float(minutes),
            }
        )
    return rows


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def predict():
        assets = load_assets()
        error = None
        results = None
        mode = "batch"
        row_index = ""

        if request.method == "POST":
            mode = request.form.get("mode", "batch")
            row_index = request.form.get("row_index", "")
            upload = request.files.get("csv_file")
            if not upload or upload.filename == "":
                error = "Please choose a CSV file to upload."
            else:
                try:
                    df = pd.read_csv(upload)
                except Exception as exc:  # pylint: disable=broad-except
                    error = f"Could not read CSV: {exc}"
                else:
                    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
                    if missing:
                        error = f"Missing columns: {', '.join(missing)}"
                    elif df.empty:
                        error = "The CSV has no rows."
                    else:
                        if mode == "single":
                            try:
                                idx = int(row_index) - 1
                            except (TypeError, ValueError):
                                error = "Row index must be a number."
                            else:
                                if idx < 0 or idx >= len(df):
                                    error = f"Row index must be between 1 and {len(df)}."
                                else:
                                    df = df.iloc[[idx]]
                        if error is None:
                            try:
                                results = run_predictions(df, assets)
                            except Exception as exc:  # pylint: disable=broad-except
                                error = str(exc)

        return render_template(
            "index.html",
            fields=FEATURE_COLUMNS,
            results=results,
            error=error,
            window=WINDOW,
            mode=mode,
            row_index=row_index,
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
