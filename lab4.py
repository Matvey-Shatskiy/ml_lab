import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import roc_curve, auc, accuracy_score

df = pd.read_csv("titanic_preprocessed.csv")

y = df["Survived"]

X = df.drop("Survived", axis=1)

X = X.select_dtypes(exclude=["object"])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=12
)

rf = RandomForestClassifier(
    n_estimators=200,
    oob_score=True,
    random_state=12
)

rf.fit(X_train, y_train)

print("Random Forest OOB accuracy:", rf.oob_score_)
print("Random Forest OOB error:", 1 - rf.oob_score_)

ada = AdaBoostClassifier(
    n_estimators=200,
    random_state=12
)

ada.fit(X_train, y_train)

gb = GradientBoostingClassifier(
    n_estimators=200,
    random_state=12
)

gb.fit(X_train, y_train)

models = {
    "Random Forest": rf,
    "AdaBoost": ada,
    "Gradient Boosting": gb
}

plt.figure(figsize=(8, 6))

for name, model in models.items():
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривые ансамблевых моделей")
plt.legend()
plt.grid(True)
plt.show()

for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} test accuracy: {acc:.3f}")