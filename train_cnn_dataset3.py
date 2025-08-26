#!/usr/bin/env python3
# Auto-generated Keras CNN trainer for your dataset.
# Usage:
#   python train_cnn_dataset3.py --csv "dataset (3).csv"
# Optional:
#   --target "label"
#   --epochs 50 --batch 32
# Saves: model.h5, label_encoder.json (if classification), metrics.json

import os, json, math, argparse, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def infer_problem_type(y: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(y):
        if y.nunique(dropna=True) > max(20, int(0.2*len(y))):
            return "regression"
    return "classification"

def best_input_shape(n_features: int):
    r = int(math.sqrt(n_features))
    if r*r == n_features:
        return ("2d", (r, r, 1))
    return ("1d", (n_features, 1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="dataset (3).csv")
    ap.add_argument("--target", type=str, default="label")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--val_split", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    assert args.target in df.columns, f"Target '{args.target}' not found. Columns: {list(df.columns)}"

    y = df[args.target]
    X_df = df.drop(columns=[args.target])

    # Prefer numeric features; fallback to one-hot for others if needed
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) == 0:
        X_df = pd.get_dummies(X_df, drop_first=False)
    else:
        X_df = X_df[num_cols]

    # Fill missing with column medians
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    prob = "classification"
    # re-infer problem type from the read data to be safe
    prob = infer_problem_type(y)

    y_enc = None
    n_classes = None
    if prob == "classification":
        # Encode labels to integers (sparse labels)
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        n_classes = int(len(le.classes_))
        with open("label_encoder.json","w",encoding="utf-8") as f:
            json.dump({"classes_": le.classes_.tolist()}, f, ensure_ascii=False, indent=2)
    else:
        # Regression
        y_enc = y.astype(float).values

    X = X_df.values.astype("float32")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    mode, shape = best_input_shape(X.shape[1])
    if mode == "2d":
        h, w, c = shape
        X = X.reshape((-1, h, w, c))
    else:
        # 1D sequence
        X = X.reshape((-1, X.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=(y_enc if prob=="classification" else None))

    # Build model
    if mode == "2d":
        inputs = keras.Input(shape=shape)
        x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
    else:
        # 1D
        inputs = keras.Input(shape=(X.shape[1], X.shape[2]))
        x = layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D()(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    if prob == "classification":
        outputs = layers.Dense(n_classes, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    else:
        outputs = layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    cb = [
        keras.callbacks.EarlyStopping(monitor=("val_accuracy" if prob=="classification" else "val_loss"),
                                      patience=8, restore_best_weights=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=cb,
        verbose=1
    )

    # Save model
    model.save("model.h5")

    # Evaluate and save metrics
    metrics = {}
    if prob == "classification":
        y_pred = model.predict(X_test).argmax(axis=1)
        acc = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics["accuracy"] = acc
        metrics["classification_report"] = report
    else:
        y_pred = model.predict(X_test).ravel()
        mse = float(mean_squared_error(y_test, y_pred))
        r2  = float(r2_score(y_test, y_pred))
        metrics["mse"] = mse
        metrics["r2"]  = r2

    with open("metrics.json","w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved: model.h5, metrics.json", "and label_encoder.json" if prob=="classification" else "")
    print(f"Problem type: {prob}  | Input mode: {mode} | Input shape: {shape}")
    print(f"Features used: {X_df.shape[1]}")

if __name__ == "__main__":
    main()
