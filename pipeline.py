from pathlib import Path
from src.data.make_dataset import load_data
from src.features.build_features import preprocess_data
from src.models.train_model import train_and_compare
from src.models.evaluate_model import evaluate
from src.explainability.shap_explain import shap_global, shap_local
from src.explainability.dice_explain import generate_counterfactuals

def main():
    # Criar pastas
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    # 1️⃣ Carregar dados
    df = load_data("data/raw/train.csv")

    # 2️⃣ Pré-processar
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    # 3️⃣ Treinar e comparar modelos
    model, results = train_and_compare(X_train, y_train, preprocessor)

    # 4️⃣ Avaliar melhor modelo
    evaluate(model, X_test, y_test)

    # 5️⃣ Explicabilidade SHAP
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = X_train.columns

    shap_global(model, X_test, feature_names)
    shap_local(model, X_test, feature_names, index=5)

    generate_counterfactuals(model, X_test, y_test, index=5, total_CFs=3)


if __name__ == "__main__":
    main()
