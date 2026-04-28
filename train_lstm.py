import json
import os

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


DB_PATH = "data/embeddings.json"
MODEL_PATH = "model/gesture_rf.joblib"
LABELS_PATH = "model/labels.json"


def load_samples(db_path):
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            "No collected samples found. Run the Streamlit app and save samples first."
        )

    with open(db_path, "r", encoding="utf-8") as f:
        db = json.load(f)

    X, y = [], []
    labels = sorted(db.keys())
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    for label, samples in db.items():
        for sample in samples:
            X.append(sample)
            y.append(label_to_idx[label])

    if len(X) < 10 or len(labels) < 2:
        raise ValueError(
            "Need at least 10 samples and 2 labels to train a useful model."
        )

    return np.array(X, dtype=np.float32), np.array(y), labels


def augment_embeddings(X, y, copies=3, noise_std=0.015, scale_jitter=0.03):
    aug_X = [X]
    aug_y = [y]

    for _ in range(copies):
        noise = np.random.normal(0.0, noise_std, size=X.shape).astype(np.float32)
        scale = np.random.uniform(1.0 - scale_jitter, 1.0 + scale_jitter, size=(X.shape[0], 1)).astype(np.float32)
        aug_sample = (X * scale) + noise
        aug_sample = np.clip(aug_sample, -2.0, 2.0)
        aug_X.append(aug_sample.astype(np.float32))
        aug_y.append(y)

    return np.vstack(aug_X), np.concatenate(aug_y)


def build_ensemble():
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
    )
    et = ExtraTreesClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
    )

    return VotingClassifier(
        estimators=[("rf", rf), ("et", et)],
        voting="soft",
        weights=[1.0, 1.2],
        n_jobs=-1,
    )


def main():
    os.makedirs("model", exist_ok=True)

    np.random.seed(42)
    X, y, labels = load_samples(DB_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_aug, y_train_aug = augment_embeddings(X_train, y_train, copies=4)
    clf = build_ensemble()
    clf.fit(X_train_aug, y_train_aug)

    y_pred = clf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))

    joblib.dump(clf, MODEL_PATH)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    print(f"Saved trained model: {MODEL_PATH}")
    print(f"Saved labels file: {LABELS_PATH}")


if __name__ == "__main__":
    main()
