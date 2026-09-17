# ============================================
# AI-Based Intrusion Detection System (NIDS)
# ============================================

import os
import numpy as np
import pandas as pd
import joblib
import warnings
from datetime import datetime

from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.utils import class_weight

warnings.filterwarnings("ignore")


# --------------------------------------------
# 1. Load Dataset
# --------------------------------------------
def load_dataset(sample_frac: float = 1.0, random_state: int = 42) -> pd.DataFrame:
    """
    Load the 10% KDD Cup 99 dataset from sklearn and optionally subsample rows.
    """
    print("[INFO] Loading KDD Cup 99 dataset...")
    data = fetch_kddcup99(as_frame=True, percent10=True)  # 10% subset built-in.[web:4]
    df = data.frame.copy()

    # Optional subsampling to speed up experimentation
    if 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
        print(f"[INFO] Subsampled dataset to {len(df)} rows (frac={sample_frac}).")

    print(f"[INFO] Dataset shape: {df.shape}")
    return df


# --------------------------------------------
# 2. Preprocess Data
# --------------------------------------------
def preprocess_data(df: pd.DataFrame):
    """
    - Convert target to binary (normal vs attack)
    - Encode categorical features
    - Return feature matrix X, label vector y, and fitted encoders
    """
    print("[INFO] Preprocessing dataset...")

    # Binary target: normal vs attack.[web:4]
    df = df.copy()
    df["target"] = df["target"].apply(
        lambda x: "normal" if x == b"normal." else "attack"
    )

    X = df.drop(columns=["target"])
    y = df["target"]

    # Track label encoders for possible future use (e.g., real-time inference)
    feature_encoders = {}

    # Encode categorical features
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        print(f"[INFO] Encoding categorical columns: {cat_cols}")
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        feature_encoders[col] = le

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)  # normal=0, attack=1 (by construction).

    # Basic sanity check on class balance (KDD is highly imbalanced).[web:4][web:9]
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"[INFO] Class distribution (0=normal, 1=attack): {class_dist}")

    return X, y, feature_encoders, label_encoder


# --------------------------------------------
# 3. Build Model Pipeline
# --------------------------------------------
def build_model(X_train, y_train):
    """
    Build a RandomForest-based pipeline with:
    - StandardScaler
    - RandomForestClassifier with tuned hyperparameters
    - Class weights computed from data to mitigate imbalance.[web:3][web:9]
    """
    print("[INFO] Building ML pipeline...")

    # Compute class weights from training distribution
    cw = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {cls: w for cls, w in zip(np.unique(y_train), cw)}
    print(f"[INFO] Computed class weights: {class_weights}")

    pipeline = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # sparse-friendly
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1
        ))
    ])

    return pipeline


# --------------------------------------------
# 4. Cross-Validation (Optional but recommended)
# --------------------------------------------
def cross_validate_model(model, X_train, y_train, folds: int = 3):
    """
    Perform stratified k-fold cross validation for more robust evaluation.[web:5][web:9]
    """
    print(f"[INFO] Running {folds}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")
    print(f"[INFO] CV F1-weighted scores: {scores}")
    print(f"[INFO] CV F1-weighted mean: {scores.mean():.4f} ± {scores.std():.4f}")


# --------------------------------------------
# 5. Train Model
# --------------------------------------------
def train_model(model, X_train, y_train):
    print("[INFO] Training model...")
    model.fit(X_train, y_train)
    return model


# --------------------------------------------
# 6. Evaluate Model
# --------------------------------------------
def evaluate_model(model, X_test, y_test, label_encoder):
    print("\n[INFO] Evaluating model...")

    y_pred = model.predict(X_test)

    # Some models may not implement predict_proba; guard accordingly
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {roc:.4f}")
    except Exception:
        print("ROC-AUC: Not available (no predict_proba).")

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    print("\nClassification Report:\n")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# --------------------------------------------
# 7. Real-Time Detection Simulation
# --------------------------------------------
def real_time_detection_simulation(model, X_test, samples: int = 10):
    print("\n[REAL-TIME INTRUSION DETECTION SIMULATION]")

    for i in range(samples):
        idx = np.random.randint(0, len(X_test))
        sample = X_test.iloc[idx:idx + 1]

        prediction = model.predict(sample)[0]
        label = "INTRUSION DETECTED" if prediction == 1 else "Normal Traffic"

        print(f"Packet #{i + 1} - Index {idx}: {label}")


# --------------------------------------------
# 8. Persist Model and Metadata
# --------------------------------------------
def save_artifacts(model, feature_encoders, label_encoder, out_dir: str = "artifacts"):
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "ai_nids_model.pkl")
    meta_path = os.path.join(out_dir, "encoders_meta.pkl")

    joblib.dump(model, model_path)
    joblib.dump(
        {
            "feature_encoders": feature_encoders,
            "label_encoder": label_encoder,
            "saved_at": datetime.utcnow().isoformat()
        },
        meta_path
    )

    print(f"\n[INFO] Model saved to: {model_path}")
    print(f"[INFO] Encoders/metadata saved to: {meta_path}")


# --------------------------------------------
# 9. Main Execution
# --------------------------------------------
def main():
    df = load_dataset(sample_frac=1.0)

    X, y, feature_encoders, label_encoder = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    model = build_model(X_train, y_train)

    # Optional: uncomment for k-fold validation before final training
    # cross_validate_model(model, X_train, y_train, folds=3)

    model = train_model(model, X_train, y_train)

    evaluate_model(model, X_test, y_test, label_encoder)

    save_artifacts(model, feature_encoders, label_encoder)

    real_time_detection_simulation(model, X_test, samples=10)


if __name__ == "__main__":
    main()
