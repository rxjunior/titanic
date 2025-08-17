from pathlib import Path
import pandas as pd

def _resolve_data_path():
    # Checa train.csv e Train.csv
    candidates = [Path("data/raw/train.csv"), Path("data/raw/Train.csv")]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Não encontrei data/raw/train.csv nem data/raw/Train.csv. "
        "Verifique o nome e o local do arquivo."
    )

def load_data(path: str | None = None) -> pd.DataFrame:
    """Carrega dataset a partir de um CSV (tenta resolver caminho automaticamente)."""
    csv_path = Path(path) if path else _resolve_data_path()
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")
    df = pd.read_csv(csv_path)
    # Sanidade mínima
    expected = {"Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colunas faltando no CSV: {missing}")
    return df
