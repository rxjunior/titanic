from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np

def train(X_train, y_train, preprocessor):
    """Treina baseline com Regressão Logística dentro do Pipeline."""
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    return model


def train_and_compare(X_train, y_train, preprocessor):
    """Treina e compara diferentes modelos"""

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    except ImportError:
        print("⚠️ XGBoost não instalado, ignorando.")

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(random_state=42)
    except ImportError:
        print("⚠️ LightGBM não instalado, ignorando.")

    results = {}
    for name, clf in models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
        results[name] = np.mean(scores)
        print(f"✅ {name} - Acurácia média (CV=5): {results[name]:.4f}")

    # Escolher melhor modelo
    best_model_name = max(results, key=results.get)
    best_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", models[best_model_name])
    ])
    best_model.fit(X_train, y_train)

    print(f"\n🏆 Melhor modelo: {best_model_name} ({results[best_model_name]:.4f})")

    return best_model, results