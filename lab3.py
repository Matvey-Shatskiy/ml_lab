import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_curve, auc

df = pd.read_csv('titanic_preprocessed.csv')

df_model = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

X_reg = df_model.drop('Age', axis=1)
y_reg = df_model['Age']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=12)

dt_reg = DecisionTreeRegressor(max_depth=3, random_state=12)
dt_reg.fit(X_train_r, y_train_r)

y_pred_r = dt_reg.predict(X_test_r)
mse = mean_squared_error(y_test_r, y_pred_r)
mae = mean_absolute_error(y_test_r, y_pred_r)

print(f"Среднеквадратичная ошибка (MSE): {mse:.4f}")
print(f"Средняя абсолютная ошибка (MAE): {mae:.4f}")

X_cls = df_model.drop('Survived', axis=1)
y_cls = df_model['Survived']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=12)

dt_cls = DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=12)
dt_cls.fit(X_train_c, y_train_c)

y_probs = dt_cls.predict_proba(X_test_c)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test_c, y_probs)
roc_auc = auc(fpr, tpr)

print(f"Площадь под ROC-кривой (ROC-AUC): {roc_auc:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-кривая (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Доля ложных срабатываний)')
plt.ylabel('True Positive Rate (Доля верных срабатываний)')
plt.title('ROC-кривая (Receiver Operating Characteristic)')
plt.legend(loc="lower right")
plt.show()