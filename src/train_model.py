# src/train_model.py
import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

from src.feature_engineering import (
    PROCESSED_PATH,
    num_features,
    cat_features,
    TARGET_DISR,
)

MODEL_PATH = Path("models/flight_risk_model.pkl")
METRICS_PATH = Path("models/flight_risk_metrics.txt")


def main():
    # 1) Charger les données pré-traitées (départs STL uniquement)
    df = pd.read_parquet(PROCESSED_PATH)

    X = df[num_features + cat_features]
    y = df[TARGET_DISR]  # NO_DISRUPTION / RISK_DISRUPTION / HIGH_DISRUPTION

    # Sanity check sur la distribution des classes
    print("Class distribution (disruption_risk):")
    print(y.value_counts(normalize=True))

    # 2) Encoder la cible en 0/1/2
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 3) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # 4) Préprocess features (numériques + catégorielles)
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    # 5) Modèle XGBoost multi-classe (disruption)
    model = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=42,
    )

    clf = Pipeline([("preprocess", preprocess), ("model", model)])

    # 6) Entraînement
    clf.fit(X_train, y_train)

    # 6b) Évaluation rapide sur le test
    y_pred = clf.predict(X_test)
    y_test_txt = le.inverse_transform(y_test)
    y_pred_txt = le.inverse_transform(y_pred)

    report = classification_report(y_test_txt, y_pred_txt)
    cm = confusion_matrix(y_test_txt, y_pred_txt, labels=le.classes_)

    print("Classification report (disruption_risk):\n", report)
    print("Confusion matrix (rows=true, cols=pred):\n", cm)

    # 7) Sauvegarder quelques métriques dans un fichier texte
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("Classes (disruption_risk): " + ", ".join(le.classes_) + "\n\n")
        f.write("Classification report:\n")
        f.write(report + "\n\n")
        f.write("Confusion matrix (rows=true, cols=pred):\n")
        f.write(pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_string())

    # 8) Sauvegarde du pipeline + label encoder + splits (pour SHAP / app)
    bundle = {
        "pipeline": clf,
        "label_encoder": le,
        "X_train": X_train,
        "X_test": X_test,
        "num_features": num_features,
        "cat_features": cat_features,
        "target_name": TARGET_DISR,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print("Model saved to", MODEL_PATH)
    print("Metrics saved to", METRICS_PATH)
    print("Classes:", le.classes_)


if __name__ == "__main__":
    main()
