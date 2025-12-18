import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)

N = 100

X = np.random.randint(0, 2, size=(N, 12))

labels = np.random.randint(0, 2, size=(N,))
Y = keras.utils.to_categorical(labels, num_classes=2)

print("Форма X:", X.shape)
print("Форма Y:", Y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)


model = keras.Sequential([
    keras.Input(shape=(12,)),
    layers.Dense(16, activation='sigmoid'),
    layers.Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Начало обучения...")
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=0
)
print("Обучение завершено.")

y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_true_classes, y_pred_classes)
print(f"\nТочность модели на тестовых данных: {accuracy:.2f}")

print("\nПримеры предсказаний:")
for i in range(min(5, len(X_test))):
    result = "Правящая партия" if y_pred_classes[i] == 0 else "Оппозиция"
    print(f"Избиратель {i+1}: Ответы {X_test[i]} -> Прогноз: {result}")

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Ошибка обучения')
plt.plot(history.history['val_loss'], label='Ошибка валидации')
plt.xlabel('Эпохи')
plt.ylabel('Loss')
plt.title('График функции потерь')
plt.legend()
plt.grid(True)
plt.show()
