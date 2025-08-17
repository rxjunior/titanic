import shap
import matplotlib.pyplot as plt
from pathlib import Path

Path("reports/figures").mkdir(parents=True, exist_ok=True)

def shap_global(pipeline, X_test, feature_names):
    """
    Explicabilidade global (importância das features)
    """
    # Pega o classificador final do pipeline
    final_model = pipeline.named_steps["classifier"]

    # Aplica pré-processamento se existir
    if "preprocessor" in pipeline.named_steps:
        X_test_transformed = pipeline.named_steps["preprocessor"].transform(X_test)
    else:
        X_test_transformed = X_test

    # Cria o explainer
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test_transformed)

    # Se binário (LightGBM retorna lista)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Plot e salvar
    shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
    plt.savefig("reports/figures/shap_global.png", bbox_inches="tight")
    plt.close()
    print("✅ SHAP global salvo em reports/figures/shap_global.png")


def shap_local(pipeline, X_test, feature_names, index=0):
    """
    Explicabilidade local (uma instância específica)
    """
    final_model = pipeline.named_steps["classifier"]

    if "preprocessor" in pipeline.named_steps:
        X_test_transformed = pipeline.named_steps["preprocessor"].transform(X_test)
    else:
        X_test_transformed = X_test

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap.initjs()
    shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        shap_values[index],
        X_test_transformed[index],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.savefig(f"reports/figures/shap_local_{index}.png", bbox_inches="tight")
    plt.close()
    print(f"✅ SHAP local salvo em reports/figures/shap_local_{index}.png")
