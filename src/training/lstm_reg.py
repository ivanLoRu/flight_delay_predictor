
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.optimizer.set_jit(True)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data/processed/dia_flights.csv"
MODELS_DIR = BASE_DIR / "models"


def load_sequences(window: int = 12):
    df = pd.read_csv(DATA_PATH)
    target = df["Departure delay (Minutes)"].values.astype(float)

    feature_df = df.drop(columns=["15min_delay", "Departure delay (Minutes)"])
    x = feature_df.values.astype(float)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaled = x_scaler.fit_transform(x)
    y_scaled = y_scaler.fit_transform(target.reshape(-1, 1))

    x_seq, y_seq = [], []
    for i in range(window, len(x_scaled)):
        x_seq.append(x_scaled[i - window : i])
        y_seq.append(y_scaled[i])

    x_seq = np.array(x_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    split = int(0.8 * len(x_seq))
    data = {
        "x_train": x_seq[:split],
        "x_test": x_seq[split:],
        "y_train": y_seq[:split],
        "y_test": y_seq[split:],
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
    }
    return data


def build_model(input_shape):
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.1),
            LSTM(32),
            Dropout(0.1),
            Dense(16, activation="relu"),
            Dense(1, activation="linear"),
        ]
    )

    model.compile(
        loss="mse",
        optimizer=Adam(learning_rate=0.001),
        metrics=["mae", "mse"],
    )
    return model


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_sequences()
    model = build_model(
        input_shape=(data["x_train"].shape[1], data["x_train"].shape[2])
    )

    model.fit(
        data["x_train"],
        data["y_train"],
        validation_split=0.1,
        epochs=20,
        batch_size=64,
        shuffle=False,
        verbose=1,
    )

    y_pred_scaled = model.predict(data["x_test"])
    y_pred = data["y_scaler"].inverse_transform(y_pred_scaled).ravel()
    y_test_true = data["y_scaler"].inverse_transform(data["y_test"]).ravel()

    mae = mean_absolute_error(y_test_true, y_pred)
    rmse = root_mean_squared_error(y_test_true, y_pred)

    print(f"MAE (minutes): {mae:.2f}")
    print(f"RMSE (minutes): {rmse:.2f}")

    model.save(MODELS_DIR / "delay_reg_lstm.keras")
    joblib.dump(data["x_scaler"], MODELS_DIR / "reg_x_scaler.pkl")
    joblib.dump(data["y_scaler"], MODELS_DIR / "reg_y_scaler.pkl")


if __name__ == "__main__":
    main()
