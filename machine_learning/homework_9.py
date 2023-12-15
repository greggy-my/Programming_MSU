from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


def calculate_Em(singular_values, m):
    """
    Calculate the characteristic E_m for a given m.

    Parameters:
    - singular_values (numpy.ndarray): Array of singular values sorted in descending order.
    - m (int): The value for m.

    Returns:
    - float: The calculated E_m value.
    """
    numerator = np.sum(singular_values[m:])  # σ_{m+1} + ... + σ_n
    denominator = np.sum(singular_values[:])  # σ_1 + ... + σ_m
    Em = numerator / denominator
    return Em


# Загрузим данные из файла (предположим, что данные хранятся в файле "data.npy")
data = np.load("../datasets/Task_9/PCA.npy")

# Применяем SVD для получения сингулярных значений
_, singular_values, _ = np.linalg.svd(data)

print(singular_values)

# Ищем минимальное m, удовлетворяющее условию E_m < 0.2
min_m = 1
while calculate_Em(singular_values, min_m) >= 0.2:
    print(calculate_Em(singular_values, min_m))
    min_m += 1

print("Min m satisfying E_m < 0.2:", min_m)


# Пересчитаем значения E_m для всех m
Em_values = [calculate_Em(singular_values, i) for i in range(1, len(singular_values) + 1)]

# Построим график
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(singular_values) + 1), Em_values, marker='o', linestyle='-')
plt.title('Graph of E_m vs m')
plt.xlabel('m')
plt.ylabel('E_m')
plt.grid(True)
plt.show()


np.random.seed(42)

mnist = fetch_openml('mnist_784', parser='auto')

X = mnist.data.to_numpy()
y = mnist.target.to_numpy()

X = X[:2000]
y = y[:2000]

# Разделение на тренировочные и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Определение оптимального количества компонент с использованием PCA и логистической регрессии
N_COMPONENTS = [1, 3, 5, 10, 15, 20, 30, 40, 50, 60]
best_accuracy = 0
best_n_components = 0

for n_components in N_COMPONENTS:
    # Применение PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Обучение логистической регрессии
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_pca, y_train)

    # Предсказание и оценка точности на тестовой выборке
    y_pred = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)

    # Обновление лучшего результата
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_components = n_components

# Вывод результата
print(f"Лучшая точность: {best_accuracy} достигнута при n_components = {best_n_components}")
