from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    """Pré-processa o dataset Titanic e retorna splits + preprocessor."""
    # Remove colunas de baixo sinal / alta cardinalidade
    X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
    y = df["Survived"]

    categorical = ["Sex", "Embarked"]
    numerical = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical),
            ("cat", categorical_transformer, categorical)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, preprocessor
