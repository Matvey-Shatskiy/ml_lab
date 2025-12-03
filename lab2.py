import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("titanic_preprocessed.csv")
df = df.select_dtypes(include=['number'])

X = df.drop(['Fare'], axis=1)
y = df['Fare']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_predict_test = linear_model.predict(X_test)

mse = mean_squared_error(y_test, y_predict_test)
mae = mean_absolute_error(y_test, y_predict_test)
rmse = root_mean_squared_error(y_test, y_predict_test)

print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

X_class = df.drop(['Survived'], axis=1)
y_class = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.4, random_state=42)

logreg_model = LogisticRegression(class_weight='balanced',max_iter=1000)
logreg_model.fit(X_train, y_train)

y_predict_test_test = logreg_model.predict(X_test)

accuracy = accuracy_score(y_test, y_predict_test_test)
print(f"Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_predict_test_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='bwr')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print(classification_report(y_test, y_predict_test_test)),