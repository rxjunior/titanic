import dice_ml
import numpy as np
import pandas as pd
from pathlib import Path

Path("reports/figures").mkdir(parents=True, exist_ok=True)

class SklearnPipelineWrapper:
    """
    Wrapper para passar o pipeline inteiro (pré-processador + classificador) para DiCE
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

def generate_counterfactuals(pipeline, X_test, y_test, index=0, total_CFs=3):
    """
    Gera contrafactuais DiCE para uma instância do dataset usando o pipeline completo
    """
    # Criar wrapper
    wrapped_model = SklearnPipelineWrapper(pipeline)

    # Instância alvo
    query_instance = X_test.iloc[[index]]

    # Criar objeto DiCE
    d = dice_ml.Data(
        dataframe=pd.concat([X_test, y_test], axis=1),
        continuous_features=list(X_test.select_dtypes(include=["float64", "int64"]).columns),
        outcome_name=y_test.name
    )
    m = dice_ml.Model(model=wrapped_model, backend="sklearn")
    exp = dice_ml.Dice(d, m)

    # Gerar contrafactuais
    cf = exp.generate_counterfactuals(query_instance, total_CFs=total_CFs, desired_class="opposite")

    if cf is not None:
        # Obter o DataFrame real
        df_cf = cf.cf_examples_list[0].final_cfs_df
        df_cf.to_csv(f"reports/figures/dice_counterfactual_{index}.csv", index=False)
        print(f"✅ Contrafactuais DiCE salvos em reports/figures/dice_counterfactual_{index}.csv")
    else:
        print(f"⚠️ Nenhum contrafactual gerado para a instância {index}. Tente outro índice ou ajuste total_CFs.")
