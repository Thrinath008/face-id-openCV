import numpy as np
import os
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_PATH = "gaze_data/gaze_data.npz"
MODEL_PATH = "gaze_data/gaze_model.pkl"


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run collect_gaze_data.py first.")

    data = np.load(DATA_PATH)
    X = data["X"]  # shape (N, 4)
    y = data["y"]

    print("Loaded data:", X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(32, 32),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=1e-3,
                    max_iter=1000,  # more iterations for stability
                    random_state=42,
                ),
            ),
        ]
    )

    print("Training MLP gaze classifier (up to 1000 iterations)...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Evaluation on held-out set:")
    print(classification_report(y_test, y_pred))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    print(f"Saved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()