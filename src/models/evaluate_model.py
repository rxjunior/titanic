from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def evaluate(model, X_test, y_test):
    """Avalia modelo com métricas + matriz de confusão."""
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\n🎯 Acurácia: {acc:.4f}\n")
    print("📊 Classification Report:\n", classification_report(y_test, preds, digits=4))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5,4))
    # Sem seaborn para evitar dependência opcional
    import numpy as np
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de Confusão")
    plt.colorbar()
    tick_marks = np.arange(2)
    classes = ["Morreu", "Sobreviveu"]
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("Real")
    plt.xlabel("Predito")
    plt.tight_layout()
    plt.show()

    return acc
