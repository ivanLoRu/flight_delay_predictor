
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
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
    target = df["15min_delay"].values
    feature_df = df.drop(columns=["15min_delay", "Departure delay (Minutes)"])
    x = feature_df.values

    x_seq, y_seq = [], []
    for i in range(window, len(x)):
        x_seq.append(x[i - window : i])
        y_seq.append(target[i])

    x_seq = np.array(x_seq)
    y_seq = np.array(y_seq)

    split = int(0.8 * len(x_seq))
    x_train = x_seq[:split]
    x_test = x_seq[split:]
    y_train = y_seq[:split]
    y_test = y_seq[split:]
    return x_train, x_test, y_train, y_test


def build_model(input_shape):
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.1),
            LSTM(32),
            Dropout(0.1),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    return model


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = load_sequences()
    model = build_model(input_shape=(x_train.shape[1], x_train.shape[2]))

    model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=4,
        batch_size=64,
        shuffle=False,
        verbose=2,
    )

    y_pred = (model.predict(x_test) > 0.5).astype(int).flatten()

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    model.save(MODELS_DIR / "delay_clf_lstm.keras")


if __name__ == "__main__":
    main()
