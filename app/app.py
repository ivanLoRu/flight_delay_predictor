from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request
from meteostat import Hourly


# Keep TensorFlow modest on thread usage for local runs
# and enable XLA where available.
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.optimizer.set_jit(True)

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data/processed/dia_flights.csv"
WINDOW = 12
STATION_ID = "72565"
CLF_THRESHOLD = 0.35

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

IMPORTANT_FEATURES: List[str] = [
    "bank_density_15min",
    "turnaround_slack",
    "dep_delay_mean3h",
    "dep_delay_mean6h",
    "dep_delay_count3h",
    "dep_delay_count6h",
    "Arrival Delay (Minutes)",
]

WEATHER_FEATURES: List[str] = [
    "temp",
    "dwpt",
    "rhum",
    "prcp",
    "wdir",
    "wspd",
    "pres",
]

MONTHLY_FEATURES: List[str] = [
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
]

DATE_COLUMNS: List[str] = [
    "date",
    "Date",
    "flight_date",
    "timestamp",
    "datetime",
]

STATIC_FEATURES: List[str] = [
    col
    for col in FEATURE_COLUMNS
    if col not in set(IMPORTANT_FEATURES + WEATHER_FEATURES + MONTHLY_FEATURES)
]

_MODEL_CACHE: Dict[str, object] = {}
_WEATHER_CACHE: Dict[str, object] = {"timestamp": None, "data": None, "error": None}


def load_assets() -> Dict[str, object]:
    """Load models and scalers once and reuse them across requests."""
    if _MODEL_CACHE:
        return _MODEL_CACHE

    paths = {
        "clf_model": MODELS_DIR / "delay_clf_lstm.keras",
        "reg_model": MODELS_DIR / "delay_reg_lstm.keras",
        "x_scaler": MODELS_DIR / "reg_x_scaler.pkl",
        "y_scaler": MODELS_DIR / "reg_y_scaler.pkl",
        "clf_x_scaler": MODELS_DIR / "clf_x_scaler.pkl",
    }
    required = ["clf_model", "reg_model", "x_scaler", "y_scaler"]
    missing = [paths[name].name for name in required if not paths[name].exists()]
    if missing:
        raise FileNotFoundError(f"Missing model assets: {', '.join(missing)}")

    assets = {
        "clf_model": tf.keras.models.load_model(paths["clf_model"]),
        "reg_model": tf.keras.models.load_model(paths["reg_model"]),
        "x_scaler": joblib.load(paths["x_scaler"]),
        "y_scaler": joblib.load(paths["y_scaler"]),
        "clf_x_scaler": joblib.load(paths["clf_x_scaler"]) if paths["clf_x_scaler"].exists() else None,
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

    clf_matrix = matrix
    if assets.get("clf_x_scaler") is not None:
        clf_matrix = assets["clf_x_scaler"].transform(matrix)
    clf_seq = build_sequences(clf_matrix)
    clf_probs = assets["clf_model"].predict(clf_seq, verbose=0).flatten()
    clf_labels = np.where(clf_probs >= CLF_THRESHOLD, "Delay", "On Time")

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


def infer_step(column: str, min_val: float, max_val: float) -> float:
    span = max_val - min_val
    if span <= 0:
        return 1.0
    if "count" in column or "density" in column or "turnaround" in column:
        return 1.0
    if span <= 5:
        return 0.1
    if span <= 50:
        return 0.5
    return round(span / 100, 2)


def load_training_stats():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

    header = pd.read_csv(DATA_PATH, nrows=0).columns
    date_col = next((col for col in DATE_COLUMNS if col in header), None)
    usecols = set(FEATURE_COLUMNS)
    if date_col:
        usecols.add(date_col)

    if date_col:
        df = pd.read_csv(DATA_PATH, usecols=list(usecols), parse_dates=[date_col])
    else:
        df = pd.read_csv(DATA_PATH, usecols=list(usecols))
    means = df[FEATURE_COLUMNS].mean(numeric_only=True).to_dict()

    ranges = {}
    for col in IMPORTANT_FEATURES:
        series = df[col]
        ranges[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
        }

    if date_col:
        grouped = df.groupby(df[date_col].dt.month)[MONTHLY_FEATURES].mean(numeric_only=True)
        monthly = {int(month): row.to_dict() for month, row in grouped.iterrows()}
        for month in range(1, 13):
            if month not in monthly:
                monthly[month] = {col: means.get(col, 0.0) for col in MONTHLY_FEATURES}
    else:
        monthly = {month: {col: means.get(col, 0.0) for col in MONTHLY_FEATURES} for month in range(1, 13)}

    return means, ranges, monthly


FEATURE_MEANS, FEATURE_RANGES, MONTHLY_AVERAGES = load_training_stats()


def get_monthly_features() -> Dict[str, float]:
    month = datetime.now().month
    data = MONTHLY_AVERAGES.get(month) or MONTHLY_AVERAGES.get(1, {})
    return {col: float(data.get(col, FEATURE_MEANS.get(col, 0.0))) for col in MONTHLY_FEATURES}


def get_current_weather() -> Dict[str, float]:
    now = datetime.utcnow()
    cached = _WEATHER_CACHE
    if cached["timestamp"] and (now - cached["timestamp"]).total_seconds() < 900:
        return cached["data"]

    end = now
    start = end - timedelta(hours=6)
    try:
        data = Hourly(STATION_ID, start, end).fetch()
        if data.empty:
            raise RuntimeError("No recent weather data returned for station 72565.")

        last = data.iloc[-1]
        weather = {}
        for col in WEATHER_FEATURES:
            val = last.get(col)
            if pd.isna(val):
                val = FEATURE_MEANS.get(col, 0.0)
            weather[col] = float(val)
    except Exception as exc:  # pylint: disable=broad-except
        # Fall back to mean weather values if Meteostat fails (e.g., pandas compatibility).
        weather = {col: float(FEATURE_MEANS.get(col, 0.0)) for col in WEATHER_FEATURES}
        cached["error"] = str(exc)

    cached["timestamp"] = now
    cached["data"] = weather
    return weather


def build_feature_row(user_values: Dict[str, float]) -> Dict[str, float]:
    row = {col: float(FEATURE_MEANS.get(col, 0.0)) for col in FEATURE_COLUMNS}
    row.update(get_monthly_features())
    row.update(get_current_weather())
    row.update(user_values)
    return row


def build_slider_configs(values: Dict[str, float]):
    sliders = []
    for idx, col in enumerate(IMPORTANT_FEATURES, start=1):
        rng = FEATURE_RANGES.get(col, {"min": 0.0, "max": 1.0})
        min_val = float(rng["min"])
        max_val = float(rng["max"])
        step = infer_step(col, min_val, max_val)
        value = float(values.get(col, FEATURE_MEANS.get(col, 0.0)))
        sliders.append(
            {
                "id": f"slider-{idx}",
                "name": col,
                "label": col,
                "min": min_val,
                "max": max_val,
                "step": step,
                "value": value,
            }
        )
    return sliders


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def predict():
        assets = load_assets()
        error = None
        results = None

        form_values = {col: float(FEATURE_MEANS.get(col, 0.0)) for col in IMPORTANT_FEATURES}

        if request.method == "POST":
            for col in IMPORTANT_FEATURES:
                raw_val = request.form.get(col, "")
                if raw_val == "":
                    continue
                try:
                    value = float(raw_val)
                except (TypeError, ValueError):
                    continue
                rng = FEATURE_RANGES.get(col)
                if rng:
                    value = max(rng["min"], min(rng["max"], value))
                form_values[col] = value

            try:
                row = build_feature_row(form_values)
                df = pd.DataFrame([row])
                results = run_predictions(df, assets)
            except Exception as exc:  # pylint: disable=broad-except
                error = str(exc)

        sliders = build_slider_configs(form_values)
        return render_template(
            "index.html",
            sliders=sliders,
            results=results,
            error=error,
            window=WINDOW,
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
